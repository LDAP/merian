# Merian Shader System

Reflection-based shader parameter passing built on Slang's type system, targeting Vulkan.

## Table of Contents

- [Concept Map](#concept-map)
- [Slang Parameter Passing Model](#slang-parameter-passing-model)
- [Slang Reflection API Reference](#slang-reflection-api-reference)
- [Merian Architecture](#merian-architecture)
- [Comparison with slang-rhi](#comparison-with-slang-rhi)

---

## Concept Map

This diagram shows how Slang's reflection concepts map to Vulkan objects
and which Slang/Merian functions bridge between them.

```
                         Slang Reflection                              Vulkan
                    ========================                    ===================

                    TypeLayoutReflection*
                    (e.g. element type of
                     ParameterBlock<T>)
                           |
          +----------------+----------------+------------------+
          |                |                |                  |
     getFieldCount()   getBindingRange   getDescriptorSet   getSubObjectRange
     getFieldByIndex()  Count/Type/...   DescriptorRange     Count/BindingRange
          |                |            Count/Type/Index          Index/Offset
          v                |            Offset(set, r)              |
  VariableLayoutReflection |                |                       |
    getName()              v                v                       v
    getOffset(UNIFORM)  BindingRanges    DescriptorRanges    SubObjectRanges
    getTypeLayout()     (abstract:       (concrete:          (ConstantBuffer /
                         "this field      "binding N in       ParameterBlock fields
                          needs a          set M with          that contain child
                          texture")        this type")         ShaderObjects)
                           |                |
                           |   getBinding   |   getDescriptorSetDescriptor
                           |   RangeFirst   |   RangeIndexOffset(0, r)
                           |   Descriptor   |          |
                           |   RangeIndex   |          |
                           |       |        |          v
                           |       +------->+    binding relative to the
                           |                     element (uint32_t)
                           v                           |
                    slang::BindingType                 v
                    (Texture, ConstantBuffer, VkDescriptorSetLayoutBinding
                     MutableRawBuffer,        { binding, type, count }
                     ParameterBlock, ...)              |
                           |                           |
                     map_slang_to_vk_                  v
                     descriptor_type()        VkDescriptorSetLayout
                           |                  (built lazily, one per
                           v                   non-empty ParameterBlock)
                    VkDescriptorType                   |
                                                       v
                                              VkPipelineLayout


    Single-element containers (ConstantBuffer<T> / ParameterBlock<T>):
    ====================================================================

    container TypeLayout (ConstantBuffer<T> / ParameterBlock<T>)
         |
         +-- getContainerVarLayout()
         |     .getOffset(DESCRIPTOR_TABLE_SLOT) --> slot of the implicit uniform
         |                                           buffer (when T has uniform data)
         |
         +-- getElementVarLayout()
               |
               +-- .getOffset(DESCRIPTOR_TABLE_SLOT) --> slot offset of T's bindings
               |                                         within the container
               +-- .getOffset(UNIFORM) --> byte offset of T's uniform data within
               |                           the container's uniform buffer
               +-- .getTypeLayout() --> T's layout, with bindings relative to T

    Slang computes all offsets; Merian never re-derives them. A binding's absolute
    slot in the ParameterBlock's descriptor set is

        element_binding_offset (per nesting level) + relative binding (from T's layout)
```

---

## Slang Parameter Passing Model

Slang has three binding modes for shader parameters. Each mode determines how data
reaches the GPU and maps to different Vulkan descriptor concepts.

### Value Binding

Scalar fields and embedded structs are **value-bound**: their data is packed into the
nearest enclosing container's uniform buffer.

```slang
struct MaterialParams {
    float roughness;   // value: offset 0 in the container's uniform buffer
    float metallic;    // value: offset 4 in the container's uniform buffer
};

struct MyParams {
    MaterialParams material;  // value binding -- embedded in MyParams' uniform data
};
```

**Vulkan mapping:** A single `VkBuffer` bound as uniform buffer in the ParameterBlock's
descriptor set (the slot is reported by `getContainerVarLayout()`). All value fields
share this buffer.

### ConstantBuffer Binding

`ConstantBuffer<T>` wraps a struct `T` in its own uniform buffer, bound as a descriptor
in the **enclosing ParameterBlock's** descriptor set. The key difference from value
binding is that `ConstantBuffer<T>` gets its own `VkBuffer` and its own descriptor
binding, rather than being packed into the enclosing uniform buffer. This is useful when
`T` is updated at a different frequency, or when `T` is large enough to warrant its own
buffer.

```slang
struct LightInfo {
    float3 direction;
    float intensity;
};

struct MyParams {
    ConstantBuffer<LightInfo> light;  // own VkBuffer, descriptor in MyParams' set
};
```

**Vulkan mapping:** A separate `VkBuffer` for `LightInfo`, written as
`VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` at a binding in the enclosing ParameterBlock's
descriptor set. The binding index is determined by Slang's layout (or overridden with
`[[vk::binding(b, s)]]`).

### ParameterBlock Binding

`ParameterBlock<T>` gives `T` its **own descriptor set**. This is the mechanism for
modular, independently-updatable parameter groups. Where a ConstantBuffer only gets its
own buffer but lives in the enclosing descriptor set, a ParameterBlock gets an entirely
separate `VkDescriptorSet` with its own `VkDescriptorSetLayout`.

```slang
struct NestedParams {
    Texture2D<float4> tex;
    float weight;
};

struct MyParams {
    ParameterBlock<NestedParams> nested;  // gets its own VkDescriptorSet
};
```

**Vulkan mapping:** `NestedParams` gets a separate `VkDescriptorSet` with its own
`VkDescriptorSetLayout`. The set index is assigned sequentially in DFS order of
ParameterBlock nesting in the SPIR-V output. ParameterBlocks whose element type has
no bindings (no uniform data, no resources) are skipped and do not consume a set index.

### Nesting Rules

These modes compose freely. A ParameterBlock can contain value fields, ConstantBuffers,
other ParameterBlocks, and resource descriptors. A ConstantBuffer can contain plain data
(scalars, vectors, structs) and other ConstantBuffers. When a ConstantBuffer is nested
inside another ConstantBuffer, the inner ConstantBuffer gets its own uniform buffer and
its descriptor is placed in the nearest enclosing ParameterBlock's descriptor set.
A ConstantBuffer cannot contain ParameterBlocks.

```slang
struct DeepParams {
    float value;                          // value in DeepParams' set
    ConstantBuffer<LightInfo> light;      // own buffer in DeepParams' set
};

struct NestedParams {
    Texture2D<float4> tex;                // resource in NestedParams' set
    float weight;                         // value in NestedParams' set
    ParameterBlock<DeepParams> deep;      // gets yet another descriptor set
    ConstantBuffer<LightInfo> light;      // own buffer in NestedParams' set
};

struct RootParams {
    MaterialParams material;              // value in RootParams' set
    ConstantBuffer<LightInfo> light;      // own buffer in RootParams' set
    ParameterBlock<NestedParams> nested;  // gets its own descriptor set
    Texture2D<float4> input;              // resource in RootParams' set
};

// Pipeline layout (3 descriptor sets, assigned in DFS order):
//   set 0: RootParams    -- material(uniform buffer at binding 0),
//                           light(uniform buffer at binding 1),
//                           input(sampled image at binding 2)
//   set 1: NestedParams  -- weight(uniform buffer at binding 0),
//                           tex(sampled image at binding 1),
//                           light(uniform buffer at binding 2)
//   set 2: DeepParams    -- value(uniform buffer at binding 0),
//                           light(uniform buffer at binding 1)
```

### Explicit Binding Annotations

Slang supports `[[vk::binding(binding, set)]]` to override auto-assigned bindings:

```slang
struct MyParams {
    [[vk::binding(1, 0)]]
    ConstantBuffer<LightInfo> light;

    [[vk::binding(2, 0)]]
    Texture2D<float4> input;
};
```

The `set` in the annotation refers to the set **relative to the parameter block context**
(typically 0 for fields within a ParameterBlock's element type). The actual Vulkan set
index is determined by where the ParameterBlock appears in the nesting hierarchy.

---

## Slang Reflection API Reference

Slang exposes a rich reflection API through several key types. All pointers are valid
as long as the `slang::IComponentType` (program) is alive.

### Program-Level Reflection

```
slang::IComponentType (program)
  --> program->getLayout() -> slang::ProgramLayout*
        +- getEntryPointCount() -> uint64_t
        +- getEntryPointByIndex(i) -> EntryPointReflection*
        +- getParameterCount() -> uint32_t          // global parameters
        +- getParameterByIndex(i) -> VariableLayoutReflection*
        +- getGlobalParamsVarLayout() -> VariableLayoutReflection*
```

`ProgramLayout` is the top-level reflection object. `getGlobalParamsVarLayout()` wraps
the whole global scope in one variable layout — when the globals contain uniform data,
its type layout has kind `ConstantBuffer` (Slang's implicit global uniform buffer),
otherwise plain `Struct`. This is what lets global and entry-point parameters share one
code path: both are just `VariableLayoutReflection`s.

### Entry Point Reflection

```
slang::EntryPointReflection*
  +- getNameOverride() -> const char*           // entry point name
  +- getStage() -> SlangStage                   // compute, vertex, fragment, ...
  +- getParameterCount() -> uint32_t            // entry point parameters
  +- getParameterByIndex(i) -> VariableLayoutReflection*
  +- getTypeLayout() -> TypeLayoutReflection*   // uniform params -> push constants
```

Each entry point parameter is a `VariableLayoutReflection` with a name and type layout.
For compute shaders, system values like `SV_DispatchThreadID` appear as parameters
alongside user-defined parameter blocks.

### Variable Layout Reflection

```
slang::VariableLayoutReflection*
  +- getName() -> const char*                          // variable name
  +- getVariable()->getName() -> const char*           // same, via Variable
  +- getTypeLayout() -> TypeLayoutReflection*          // type information
  +- getOffset(SLANG_PARAMETER_CATEGORY_UNIFORM) -> size_t  // byte offset in uniform buffer
  +- getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT) -> size_t // binding offset
```

Offsets are always relative to the enclosing container. For the element variable layout
of a `ConstantBuffer<T>`/`ParameterBlock<T>` they describe where `T`'s data starts
within the container (e.g. descriptor slot 1 when the implicit uniform buffer occupies
slot 0).

### Type Layout Reflection

This is the most important reflection type. It describes a Slang type's layout,
including fields, binding ranges, descriptor sets, and sub-objects.

#### Identity and Size

```
slang::TypeLayoutReflection*
  +- getName() -> const char*                               // type name ("MyStruct")
  +- getKind() -> TypeReflection::Kind                      // Struct, ParameterBlock, ...
  +- getType()->getKind() -> TypeReflection::Kind           // same via TypeReflection
  +- getSize(SLANG_PARAMETER_CATEGORY_UNIFORM) -> size_t    // ordinary data size in bytes
```

`getSize(SLANG_PARAMETER_CATEGORY_UNIFORM)` returns the total size of value-bound data
for this type. This is the size of the container's uniform buffer. If the type has no
value fields (e.g., a struct containing only textures), this returns 0 and no uniform
buffer is needed.

`TypeReflection::Kind` values and their Vulkan meaning:

| Kind | Vulkan Concept |
|------|----------------|
| `Struct` | Plain data struct, fields packed into uniform buffer |
| `Scalar` | Single value (float, int, etc.) in uniform buffer |
| `Vector` | Vector (float3, etc.) in uniform buffer |
| `Matrix` | Matrix in uniform buffer |
| `Array` | Array, may be uniform data or resource array |
| `Resource` | Texture, buffer -- descriptor binding |
| `SamplerState` | Sampler -- descriptor binding |
| `ConstantBuffer` | Own uniform buffer, descriptor in enclosing ParameterBlock's set |
| `ParameterBlock` | Own descriptor set |

#### Field Navigation

```
  +- getFieldCount() -> uint32_t
  +- getFieldByIndex(i) -> VariableLayoutReflection*
  +- findFieldIndexByName(name) -> SlangInt          // -1 if not found
  +- getFieldBindingRangeOffset(i) -> uint32_t       // binding range index for field i
```

`getFieldBindingRangeOffset(field_index)` maps a field index to its first binding range
index. This is the binding range where the field's descriptor(s) start.

**Important caveat:** Value-only fields (scalars, vectors, plain structs) that only
contribute to the uniform data buffer share the same binding range offset as the next
resource or ConstantBuffer or ParameterBlock field. For example, if field 0 is a
`float` (value) and field 1 is a `ConstantBuffer<T>`, both will have
`getFieldBindingRangeOffset()` return the same value. You must check the field's Kind
(via `getFieldByIndex(i)->getTypeLayout()->getKind()`) to distinguish them.

#### Array Navigation

```
  +- getElementCount() -> size_t                              // array length
  +- getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM) -> size_t  // stride in bytes
  +- getElementTypeLayout() -> TypeLayoutReflection*          // element type
```

#### Single-Element Containers

`ConstantBuffer<T>` and `ParameterBlock<T>` type layouts are unwrapped through their
variable layouts, never through bare type layouts, so that Slang's offsets are used:

```
  +- getElementVarLayout() -> VariableLayoutReflection*
  |    element type layout (bindings relative to T) + offsets of T within the container
  +- getContainerVarLayout() -> VariableLayoutReflection*
       offsets of the container's own resources (the implicit uniform buffer)
```

The same unwrapping applies to both ConstantBuffer and ParameterBlock. Do **not**
create element layouts for other binding ranges (RWStructuredBuffer, Texture, ...):
these unwrap to scalar element types and would recurse infinitely.

#### Binding Ranges

A binding range represents a contiguous group of descriptors of the same type. Each
resource, ConstantBuffer, or ParameterBlock field contributes one or more binding
ranges. Value-only fields do not have their own binding range -- they are implicitly
part of the container's uniform buffer.

```
  +- getBindingRangeCount() -> uint32_t
  +- getBindingRangeType(br) -> slang::BindingType
  +- getBindingRangeBindingCount(br) -> uint32_t              // descriptor count
  +- getBindingRangeLeafTypeLayout(br) -> TypeLayoutReflection*
  +- getBindingRangeLeafVariable(br) -> VariableReflection*
  +- getBindingRangeFirstDescriptorRangeIndex(br) -> SlangInt // maps to descriptor range
```

**`getBindingRangeType(binding_range_index)`** returns what kind of descriptor this
binding range represents (texture, uniform buffer, storage image, etc.).

**`getBindingRangeBindingCount(binding_range_index)`** returns how many descriptors are
in this range. For a single texture field this is 1. For an array of 4 textures this
is 4. `SLANG_UNKNOWN_SIZE` indicates an array sized by a link-time constant
(`extern static const`); resolve it via
`getBindingRangeLeafVariable(br)->getType()->getElementCount(program_layout)`.

**`getBindingRangeFirstDescriptorRangeIndex(binding_range_index)`** is the bridge
between binding ranges and descriptor ranges. A binding range is Slang's abstract
concept ("this field needs a texture descriptor"), while a descriptor range is the
concrete layout ("this descriptor goes at relative binding N"). It returns -1 for
ParameterBlock ranges, which have no descriptors in the enclosing set.

`slang::BindingType` values and their Vulkan mapping:

| BindingType | Vulkan Descriptor Type |
|-------------|----------------------|
| `Texture` | `eSampledImage` |
| `MutableTexture` | `eStorageImage` |
| `ConstantBuffer` | `eUniformBuffer` |
| `RawBuffer` / `MutableRawBuffer` | `eStorageBuffer` |
| `Sampler` | `eSampler` |
| `CombinedTextureSampler` | `eCombinedImageSampler` |
| `RayTracingAccelerationStructure` | `eAccelerationStructureKHR` |
| `ParameterBlock` | No descriptor -- represents a sub-object with its own set |

#### Descriptor Set Ranges

These give the binding indices within a descriptor set, relative to the type layout
they are queried from:

```
  +- getDescriptorSetCount() -> uint32_t
  +- getDescriptorSetDescriptorRangeCount(set) -> uint32_t
  +- getDescriptorSetDescriptorRangeType(set, r) -> slang::BindingType
  +- getDescriptorSetDescriptorRangeDescriptorCount(set, r) -> uint32_t
  +- getDescriptorSetDescriptorRangeIndexOffset(set, r) -> uint32_t   // relative binding
```

**How to get the relative binding number from a binding range:**

```cpp
// Step 1: binding range -> descriptor range (abstract -> concrete)
SlangInt first_dr = type_layout->getBindingRangeFirstDescriptorRangeIndex(br);

// Step 2: descriptor range -> binding number relative to this type layout
uint32_t binding = type_layout->getDescriptorSetDescriptorRangeIndexOffset(0, first_dr);
```

The absolute Vulkan binding is `element_binding_offset + binding`, where the offset
comes from the container's `getElementVarLayout()`.

#### Sub-Object Ranges

```
  +- getSubObjectRangeCount() -> uint32_t
  +- getSubObjectRangeBindingRangeIndex(i) -> uint32_t  // which binding range
  +- getSubObjectRangeOffset(i) -> VariableLayoutReflection*  // field offsets
```

Sub-object ranges identify binding ranges that contain ConstantBuffer or ParameterBlock
sub-objects. These are the fields where the parent type "points to" a child object,
as opposed to embedding data directly.

`getSubObjectRangeOffset(i)->getOffset(DESCRIPTOR_TABLE_SLOT)` is the field's
descriptor slot offset within the struct. Combined with the sub-object's own
container/element var layout offsets this yields, for a ConstantBuffer field:

```
uniform buffer descriptor slot = field offset + container var layout offset
element bindings start at      = field offset + element var layout offset
```

---

## Merian Architecture

### Overview

```
SlangProgram --> SlangProgramEntryPoint --> ShaderObject --> ShaderCursor
     |                   |                       |
     |            (pipeline layout,        (ParameterBlock / ConstantBuffer /
     |             set index tree)          Struct objects, see below)
     |
ShaderObjectLayout --- cached per program via
                       SlangProgram::get_or_create_object_layout
```

A `ParameterBlock<T>` produces **two** ShaderObjects: a container object that owns the
Vulkan resources and an element object for `T` that records writes. The same applies to
`ConstantBuffer<T>`.

### Class Responsibilities

**`SlangProgram`** -- Wraps a compiled Slang program. Provides access to program
reflection (`ProgramLayout*`) and SPIR-V binary, and owns the per-program
`ShaderObjectLayout` cache (`get_or_create_object_layout`). Layouts are deduplicated
per program — not globally — because link-time constants make binding layouts
program-dependent. Keeps the Slang session alive so all reflection pointers remain
valid.

**`SlangProgramEntryPoint`** -- Represents one entry point. Builds and caches the
pipeline layout and the descriptor set index tree (`NestedParameterBlockInfo`). Global
and entry-point parameters share the same machinery: the global scope is treated as one
ParameterBlock-like container built from `getGlobalParamsVarLayout()`.

**`ShaderObjectLayout`** -- Cached reflection data for one Slang type; its kind decides
the role:

- *Struct*: binding ranges with **relative** bindings, sub-object ranges (whose layouts
  are container layouts), uniform size, resource slot count. Never builds a
  `VkDescriptorSetLayout`.
- *ConstantBuffer*: element layout + the offsets Slang assigned to the element
  (`getElementVarLayout()` / `getContainerVarLayout()`).
- *ParameterBlock*: same as ConstantBuffer, plus the `VkDescriptorSetLayout` — built
  **lazily** on first use. `has_bindings()` answers from reflection whether the block
  contributes a Vulkan set at all, without creating Vulkan objects.

**`ShaderObject`** -- The runtime parameter container; behavior follows the layout kind:

- *ParameterBlock objects* own the `DescriptorStorage` (cached descriptor writes,
  replayed to newly allocated sets), the live-set registry, the uniform buffer for the
  element's value data, and the element object.
- *ConstantBuffer objects* own their uniform buffer and the element object. Their
  uniform-buffer descriptor is written into every ParameterBlock they are attached to.
- *Struct objects* own no Vulkan resources. Uniform writes go directly into the owning
  container's staging memory. Descriptor writes are recorded in a relative slot record
  and forwarded to all attached descriptor targets `{ParameterBlock, binding_base}`;
  attaching replays the whole record. A ParameterBlock element has exactly one fixed
  target (its block); a ConstantBuffer element gains and loses targets as the
  ConstantBuffer is set into (or removed from) ParameterBlocks — this also covers
  ConstantBuffer-in-ConstantBuffer and ConstantBuffers shared across ParameterBlocks.

**`ShaderCursor`** -- Lightweight `(ShaderObject*, TypeLayout*, ShaderOffset)` tuple for
navigating the parameter space of a struct object. Supports
`cursor["field"]["nested_field"] = value` syntax. Navigating to a ConstantBuffer or
ParameterBlock field auto-creates the container sub-object (which creates its element)
and dereferences into the element. `get_cursor()` on a container returns the element's
cursor, so the container/element split is invisible at the call site.

**`ShaderObjectAllocator`** -- Abstract allocator for descriptor sets.
`FrameCachingShaderObjectAllocator` caches sets per `(object, frame_index)` to avoid
re-allocation every frame. Call `set_iteration(frame_index)` before binding to select
the current frame's cache slot.

### Descriptor Write Model: Update-Time, Not Bind-Time

All descriptor writes happen eagerly when the user sets a value (via cursor or
`set_subobject`), NOT at bind time. A steady-state frame where nothing changed requires
zero descriptor writes.

```
Update time (cursor assignment):                   Bind time (dispatch):
================================                    ====================

cursor["tex"] = texture;                            bind_as_parameter_block()
  |                                                   |
  +-> record in slot record (replay)                  +-> allocate set
  +-> for each attached ParameterBlock:               +-> if new: replay from storage
        queue write into its storage + live sets      +-> upload dirty uniform staging
                                                      +-> set->update() + bind()
cursor["light"]["intensity"] = 1.5f;
  |                                                  (zero descriptor writes if nothing
  +-> auto-creates the ConstantBuffer object,         changed and the set was already
      writes its uniform-buffer descriptor into       tracked)
      the owning ParameterBlock's storage/sets,
      attaches its element
```

### Binding Flow

Here is the complete flow for a 3-level deep parameter hierarchy:

```cpp
// Setup (once)
auto entry_point = SlangProgramEntryPoint::create(program, "main");
auto pipeline_layout = entry_point->get_pipeline_layout(context);
auto pipeline = ComputePipeline::create(pipeline_layout, entry_point->specialize());
auto params = entry_point->create_shader_object(context, "params", allocator);

// Write parameters (cursor auto-creates sub-objects)
auto cursor = params->get_cursor();
cursor["material"]["roughness"] = 0.8f;                // value in set 0
cursor["light"]["intensity"] = 1.5f;                    // ConstantBuffer in set 0
cursor["nested"]["weight"] = 0.3f;                      // value in set 1
cursor["nested"]["tex"] = my_texture;                   // resource in set 1
cursor["nested"]["deep"]["value"] = 42.0f;              // value in set 2

// Bind and dispatch (every frame)
allocator->set_iteration(frame_index);
cmd->bind(pipeline);
entry_point->bind_entry_point_parameter("params", params, cmd, pipeline, obj_allocator);
cmd->dispatch(extent, 16, 16);
```

The bind call triggers this chain:

```
bind_entry_point_parameter("params", params, ...)
  |
  +- if params->has_pending_uploads(): transfer barrier
  |
  +- params->bind_as_parameter_block(cmd, pipeline, set=0)
  |    +- if uploads pending: upload dirty uniform staging
  |    |    (own + ConstantBuffer sub-objects, recursively)
  |    +- obj_allocator->allocate(params) -> {descriptor set for set 0, freshly_allocated}
  |    +- if freshly allocated: descriptors->replay_to(set)
  |    +- set->update() + set->bind(cmd, pipeline, 0)
  |
  +- bind_nested_parameter_blocks(params, set index tree)
  |    |
  |    +- nested->bind_as_parameter_block(cmd, pipeline, set=1)
  |    +- bind_nested_parameter_blocks(nested, children)
  |         |
  |         +- deep->bind_as_parameter_block(cmd, pipeline, set=2)
  |
  +- if uploads happened: transfer barrier
```

**Key design point:** descriptor writes (light's uniform buffer, nested's texture, ...)
were already written to the ParameterBlock's descriptor storage when the cursor
assignment happened. Uniform writes flag every affected ParameterBlock
(`uploads_pending`), so a clean frame binds without any transfer barriers, uniform
walks, or descriptor writes — just the allocator lookup and
`vkCmdBindDescriptorSets`. Upload barriers cover all commands (not the binding
pipeline's stages) because shared objects can be read by other pipelines later.

### Descriptor Set Index Assignment

Vulkan descriptor set indices are assigned sequentially: first the global scope (if it
has bindings), then ParameterBlocks in DFS order of nesting, **skipping blocks whose
element type has no bindings**. This matches Slang's SPIR-V output. The assignment
happens in `get_pipeline_layout()`; the result is cached in a tree of
`NestedParameterBlockInfo{subobject_range_index, set_index, children}` used at bind
time. Blocks with `set_index == NO_DESCRIPTOR_SET` are skipped; their descriptor set
layout is never built (emptiness is answered by `ShaderObjectLayout::has_bindings()`
from reflection alone).

### Write Propagation

Uniform writes copy into the owning container's staging memory and mark it dirty; the
GPU upload happens at bind time (only if dirty). A freshly created container uploads its
zero-initialized staging once, so unwritten fields read as zero instead of garbage.

Resource writes are recorded in the struct object's slot record and applied to every
attached ParameterBlock's `DescriptorStorage` and live sets. Attaching a struct to a new
ParameterBlock (directly at element creation, or transitively when a ConstantBuffer is
set into a block) replays the slot record at the accumulated binding offset and recurses
into ConstantBuffer sub-objects. Reassigning a sub-object detaches the old object's
element and replays the new one over the same bindings.

A `ConstantBuffer<T>`/`ParameterBlock<T>` field is always present in the layout and must
stay bound to a fully-populated object; assigning `nullptr` is not supported. Reassignment
to another fully-written object of the same type is fine: the replacement writes every
binding the old object did (resources and the implicit uniform-buffer descriptor), so no
stale descriptor survives. A binding the replacement leaves unwritten would be an unbound
descriptor regardless — populate every resource the type declares.

### Resource-Only ParameterBlocks

A ParameterBlock whose element type has no value fields (e.g., a struct containing only
textures) is valid and works correctly. The uniform size will be 0, so no uniform buffer
is allocated. Only the resource descriptors are written.

### Debug Formatting

All key classes have `format_*` methods that print reflection information:

```cpp
SPDLOG_INFO("{}", program->format_reflection());
SPDLOG_INFO("{}", entry_point->format_reflection(context));
SPDLOG_INFO("{}", format_as(*shader_object));
SPDLOG_INFO("{}", format_as(cursor));
SPDLOG_INFO("{}", format_type_layout(type_layout));  // standalone, in slang_utils
```

---

## Comparison with slang-rhi

[slang-rhi](https://github.com/shader-slang/slang-rhi) is Slang's official
cross-platform hardware interface. Merian's shader system is a Vulkan-specific
implementation inspired by slang-rhi but with different design choices.

### Architecture Differences

| Aspect | slang-rhi | Merian |
|--------|-----------|--------|
| **Backend** | Cross-platform (Vulkan, D3D12, Metal, CUDA, WGPU) | Vulkan only |
| **Descriptor management** | Deferred: all writes happen at bind time via `BindingDataBuilder` | Eager: writes cached in slot records / `DescriptorStorage`, replayed on attach |
| **Uniform data** | CPU byte buffer (`m_data`), bulk-copied to pooled buffer at bind time | Staging per container, uploaded via `StagingMemoryManager` when dirty |
| **Container model** | One `ShaderObject` per `T`, container offsets applied at bind time | Two ShaderObjects per `ConstantBuffer<T>`/`ParameterBlock<T>`: container + element |
| **Resource storage** | Flat `m_slots` array of `ResourceSlot` tagged unions | Flat slot record (`std::variant`) per struct object |
| **Pipeline layout** | `RootShaderObjectLayout` owns `VkPipelineLayout` + all set layouts | `SlangProgramEntryPoint` builds and caches pipeline layout |
| **Push constants** | Built-in: entry point data routed to push constants | Range reserved in pipeline layout; not used by the object system |
| **Specialization** | Built-in: existential/interface types with RTTI + witness tables | Not yet supported |
| **Caching** | Version-based (`m_uid` + `m_version`) with `finalize()` | Frame-based via `FrameCachingShaderObjectAllocator` |
| **Parameter navigation** | `ShaderCursor` (library-provided, stateless) | `ShaderCursor` (similar, but auto-creates sub-objects) |

### What slang-rhi Has That Merian Doesn't (Yet)

- **Existential/interface type** support with RTTI headers and witness tables
- **Specialization** caching for generic shader objects
- **Push constant** routing for entry-point parameters
- **Finalization** (`finalize()`) for immutable, reusable shader objects
- **Cross-backend** support (D3D12, Metal, CUDA, WGPU)
