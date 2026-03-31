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
    getTypeLayout()     (abstract:       (concrete:          (CB/PB fields that
                         "this field      "binding N in       contain child
                          needs a          set M with          ShaderObjects)
                          texture")        this type")
                           |                |
                           |   getBinding   |   getDescriptorSetDescriptor
                           |   RangeFirst   |   RangeIndexOffset(0, r)
                           |   Descriptor   |          |
                           |   RangeIndex   |          |
                           |       |        |          v
                           |       +------->+    Vulkan binding
                           |                     number (uint32_t)
                           v                           |
                    slang::BindingType                 v
                    (Texture, CB,           VkDescriptorSetLayoutBinding
                     MutableRawBuffer,        { binding, type, count }
                     ParameterBlock, ...)              |
                           |                           |
                     map_slang_to_vk_                  v
                     descriptor_type()        VkDescriptorSetLayout
                           |                  (DescriptorSetLayout)
                           v                           |
                    VkDescriptorType                   v
                                              VkPipelineLayout
                                              (one set layout per PB,
                                               empty PBs skipped)


    Leaf type unwrapping for sub-objects (SlangObjectLayout constructor):
    =====================================================================

    getBindingRangeLeafTypeLayout(br)
         |
         v
    TypeLayoutReflection* (leaf)
         |
         +-- ParameterBlock:  getElementTypeLayout()
         |   (preserves descriptor set context for correct binding offsets)
         |
         +-- ConstantBuffer:  getElementVarLayout()->getTypeLayout()
             (gives clean struct layout without flattened CB descriptor ranges)


    CB binding offset computation (set_subobject / compute_cb_binding_deltas):
    ===========================================================================

    parent TypeLayout
         |
         +-- getSubObjectRangeOffset(sor) --> VariableLayoutReflection*
         |     .getOffset(DESCRIPTOR_TABLE_SLOT) --> sub_range_offset
         |
         +-- getBindingRangeLeafTypeLayout(br) --> ConstantBuffer<T> layout
               |
               +-- getContainerVarLayout()
               |     .getOffset(DESCRIPTOR_TABLE_SLOT) --> container_offset (UBO binding)
               |
               +-- getElementVarLayout()
                     .getOffset(DESCRIPTOR_TABLE_SLOT) --> element_offset (T's content)

    absolute_ubo_binding     = base_binding + sub_range_offset + container_offset
    absolute_element_binding = base_binding + sub_range_offset + element_offset
```

---

## Slang Parameter Passing Model

Slang has three binding modes for shader parameters. Each mode determines how data
reaches the GPU and maps to different Vulkan descriptor concepts.

### Value Binding

Scalar fields and embedded structs are **value-bound**: their data is packed into the
parent's ordinary data buffer (a `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` at binding 0 of
the parent's descriptor set).

```slang
struct MaterialParams {
    float roughness;   // value: offset 0 in parent's uniform buffer
    float metallic;    // value: offset 4 in parent's uniform buffer
};

struct MyParams {
    MaterialParams material;  // value binding -- embedded in MyParams' uniform buffer
};
```

**Vulkan mapping:** A single `VkBuffer` bound as uniform buffer at binding 0 of the
parameter block's descriptor set. All value fields share this buffer.

### ConstantBuffer Binding

`ConstantBuffer<T>` wraps a struct `T` in its own uniform buffer, bound as a descriptor
in the **parent's** descriptor set. The key difference from value binding is that
`ConstantBuffer<T>` gets its own `VkBuffer` and its own descriptor binding, rather than
being packed into the parent's ordinary data buffer. This is useful when `T` is updated
at a different frequency than the parent, or when `T` is large enough to warrant its
own buffer.

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
`VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` at a binding in the parent ParameterBlock's
descriptor set. The binding index is determined by Slang's layout (or overridden with
`[[vk::binding(b, s)]]`).

### ParameterBlock Binding

`ParameterBlock<T>` gives `T` its **own descriptor set**. This is the mechanism for
modular, independently-updatable parameter groups. Where a ConstantBuffer only gets its
own buffer but lives in the parent's descriptor set, a ParameterBlock gets an entirely
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

ConstantBuffer nesting example:

```slang
struct InnerCB {
    float inner_val;
};

struct OuterCB {
    float outer_val;
    ConstantBuffer<InnerCB> inner;    // nested CB gets its own uniform buffer
};

struct MyParams {
    ConstantBuffer<OuterCB> cb_in_cb; // OuterCB buffer at binding N,
                                      // InnerCB buffer at binding N+1,
                                      // both in MyParams' descriptor set
};
```

When a ConstantBuffer contains another ConstantBuffer, both uniform buffer descriptors
are placed in the nearest enclosing ParameterBlock's descriptor set. The Slang reflection
API provides the correct binding offsets via `getContainerVarLayout()` and
`getElementVarLayout()` on the leaf type layout (see [Concept Map](#concept-map)).
Descriptor writes for nested CBs are propagated to the owning PB at update time (when
`set_subobject` is called), not at bind time.

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
```

`ProgramLayout` is the top-level reflection object. It gives access to entry points
and global (module-scope) parameters.

### Entry Point Reflection

```
slang::EntryPointReflection*
  +- getNameOverride() -> const char*           // entry point name
  +- getStage() -> SlangStage                   // compute, vertex, fragment, ...
  +- getParameterCount() -> uint32_t            // entry point parameters
  +- getParameterByIndex(i) -> VariableLayoutReflection*
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
  +- getBindingSpace() -> uint32_t                     // space within parent context
```

`getOffset(SLANG_PARAMETER_CATEGORY_UNIFORM)` returns the byte offset within the
parent's ordinary data buffer. For ConstantBuffer and ParameterBlock fields this
returns 0 because they don't contribute to the parent's uniform data -- they have their
own storage.

`getBindingSpace()` returns the binding space relative to the parent context. For
nested ParameterBlocks this is always 0 (it is NOT the absolute Vulkan set index).
The actual Vulkan set index must be computed by walking the ParameterBlock nesting
hierarchy.

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
for this type. This is the size of the uniform buffer needed at binding 0 of the
descriptor set. If the type has no value fields (e.g., a struct containing only
textures), this returns 0 and no uniform buffer is needed.

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
| `ConstantBuffer` | Own uniform buffer, descriptor in parent's set |
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

`getElementTypeLayout()` also unwraps `ConstantBuffer<T>` and `ParameterBlock<T>` to
get the inner type `T`'s layout. For ParameterBlock types this preserves descriptor set
context (binding offsets account for the PB's implicit UBO). For ConstantBuffer types,
prefer `getElementVarLayout()->getTypeLayout()` instead, which gives a clean struct
layout without flattened CB descriptor ranges that can cause duplicate bindings at
binding 0.

#### Binding Ranges

A binding range represents a contiguous group of descriptors of the same type. Each
resource, ConstantBuffer, or ParameterBlock field contributes one or more binding
ranges. Value-only fields do not have their own binding range -- they are implicitly
part of the uniform buffer at binding 0.

```
  +- getBindingRangeCount() -> uint32_t
  +- getBindingRangeType(br) -> slang::BindingType
  +- getBindingRangeBindingCount(br) -> uint32_t              // descriptor count
  +- getBindingRangeLeafTypeLayout(br) -> TypeLayoutReflection*
  +- getBindingRangeFirstDescriptorRangeIndex(br) -> SlangInt // maps to descriptor range
```

**`getBindingRangeType(binding_range_index)`** returns what kind of descriptor this
binding range represents (texture, uniform buffer, storage image, etc.).

**`getBindingRangeBindingCount(binding_range_index)`** returns how many descriptors are
in this range. For a single texture field this is 1. For an array of 4 textures this
is 4.

**`getBindingRangeLeafTypeLayout(binding_range_index)`** returns the type layout of the
"leaf" type for this binding range. For a `ConstantBuffer<T>` field this returns the
`ConstantBuffer<T>` type layout (not `T`), from which you can get the container and
element var layouts for binding offset computation.

**`getBindingRangeFirstDescriptorRangeIndex(binding_range_index)`** is the bridge
between binding ranges and descriptor ranges. A binding range is Slang's abstract
concept ("this field needs a texture descriptor"), while a descriptor range is the
concrete Vulkan layout ("this descriptor goes at binding N in descriptor set M").
This method returns the index into the descriptor range array where this binding range's
descriptors start. You then use that index with
`getDescriptorSetDescriptorRangeIndexOffset()` to get the actual Vulkan binding number.

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

ParameterBlock binding ranges don't have descriptors in set 0. They represent
sub-objects that get their own descriptor sets. The binding range exists so that Slang
can track the sub-object, but no actual Vulkan descriptor is written for it.

#### Descriptor Set Ranges

These give the actual Vulkan binding indices within a descriptor set:

```
  +- getDescriptorSetCount() -> uint32_t
  +- getDescriptorSetDescriptorRangeCount(set) -> uint32_t
  +- getDescriptorSetDescriptorRangeType(set, r) -> slang::BindingType
  +- getDescriptorSetDescriptorRangeDescriptorCount(set, r) -> uint32_t
  +- getDescriptorSetDescriptorRangeIndexOffset(set, r) -> uint32_t   // Vulkan binding
```

For a ParameterBlock's element type, set 0 contains all resource bindings for that
ParameterBlock. `getDescriptorSetDescriptorRangeIndexOffset(0, range_index)` gives the
Vulkan binding index for that descriptor range.

**How to get the Vulkan binding number from a binding range:**

Given a binding range index `br`, you need two steps:

1. Get the descriptor range index: `first_dr = type_layout->getBindingRangeFirstDescriptorRangeIndex(br)`
2. Get the Vulkan binding from that descriptor range: `vk_binding = type_layout->getDescriptorSetDescriptorRangeIndexOffset(0, first_dr)`

```cpp
// Step 1: binding range -> descriptor range (abstract -> concrete)
SlangInt first_dr = type_layout->getBindingRangeFirstDescriptorRangeIndex(br);

// Step 2: descriptor range -> Vulkan binding number
uint32_t vk_binding = type_layout->getDescriptorSetDescriptorRangeIndexOffset(0, first_dr);
```

#### Sub-Object Ranges

```
  +- getSubObjectRangeCount() -> uint32_t
  +- getSubObjectRangeBindingRangeIndex(i) -> uint32_t  // which binding range
  +- getSubObjectRangeOffset(i) -> VariableLayoutReflection*  // offset for CB binding calc
```

Sub-object ranges identify binding ranges that contain ConstantBuffer or ParameterBlock
sub-objects. These are the fields where the parent type "points to" a child object,
as opposed to embedding data directly.

`getSubObjectRangeOffset(i)` returns a `VariableLayoutReflection*` that provides the
descriptor table slot offset for computing Vulkan bindings of nested ConstantBuffers.
Combined with the leaf type layout's `getContainerVarLayout()` and
`getElementVarLayout()`, this enables the full binding offset calculation shown in the
[Concept Map](#concept-map).

#### Leaf Type Unwrapping

When building `SlangObjectLayout` for sub-object ranges, the leaf type must be
unwrapped to get the element type layout. The correct unwrapping method depends on the
binding type:

- **ParameterBlock**: Use `getElementTypeLayout()` on the leaf type. This preserves the
  PB's descriptor set context -- the resulting type layout's descriptor ranges have
  correct binding offsets that account for the implicit UBO at binding 0. Using
  `getElementVarLayout()->getTypeLayout()` instead would lose this context and produce
  a raw struct layout missing resource descriptors.

- **ConstantBuffer**: Use `getElementVarLayout()->getTypeLayout()` on the leaf type.
  This gives a clean struct layout for `T`. Using `getElementTypeLayout()` for CBs
  produces a type layout with a ConstantBuffer descriptor range at binding 0 that
  overlaps with the implicit UBO for uniform data.

- **Other types** (RWStructuredBuffer, Texture, etc.): Do **not** create element
  layouts. These types unwrap to scalar element types and would cause infinite recursion
  in the layout constructor.

### Complete Reflection Walk Example

Given this shader:

```slang
struct LightInfo {
    float3 direction;
    float intensity;
};

struct NestedParams {
    Texture2D<float4> tex;
    float weight;
};

struct MyParams {
    float value;                           // field 0: scalar -> value binding
    ConstantBuffer<LightInfo> light;       // field 1: ConstantBuffer -> own buffer
    ParameterBlock<NestedParams> nested;   // field 2: ParameterBlock -> own set
    Texture2D<float4> tex;                 // field 3: resource -> texture descriptor
};
```

The reflection for `MyParams` (as the element type of a ParameterBlock) produces:

```
TypeLayout 'MyParams' (uniform_size=16)
  fields (4):
    [0] 'value':  kind=Scalar,          uniform_offset=0, binding_range_offset=0
    [1] 'light':  kind=ConstantBuffer,  uniform_offset=0, binding_range_offset=0
    [2] 'nested': kind=ParameterBlock,  uniform_offset=0, binding_range_offset=1
    [3] 'tex':    kind=Resource,        uniform_offset=0, binding_range_offset=2
  binding_ranges (3):
    [0] type=ConstantBuffer,  count=1   -> descriptor range 0 -> vk_binding=1
    [1] type=ParameterBlock,  count=1   -> no descriptor (sub-object only)
    [2] type=Texture,         count=1   -> descriptor range 1 -> vk_binding=2
  descriptor_sets (1):
    set 0:
      range [0]: type=ConstantBuffer, count=1, vk_binding=1
      range [1]: type=Texture,        count=1, vk_binding=2
  subobject_ranges (2):
    [0] binding_range=0  (ConstantBuffer)
    [1] binding_range=1  (ParameterBlock)
```

Note the following key details:

1. **`value` (field 0) and `light` (field 1) share `binding_range_offset=0`.**
   The `float value` field doesn't need its own binding range because it goes into the
   uniform buffer at binding 0. Its `binding_range_offset` points to the next real
   binding range (the ConstantBuffer). To distinguish them, check the field's Kind.

2. **`uniform_size=16` bytes** because only the `float value` field (4 bytes, padded
   to 16 for std140 layout) contributes to the ordinary data buffer. ConstantBuffer
   and ParameterBlock fields have their own storage.

3. **Binding 0 is implicit.** The uniform buffer for `MyParams`' own value data is
   always at binding 0. It doesn't appear in the binding ranges -- it's implicit. The
   first explicit binding range (ConstantBuffer for `light`) maps to `vk_binding=1`.

4. **ParameterBlock has no descriptor range.** Binding range 1 (ParameterBlock) doesn't
   generate a descriptor in set 0. It exists only so Slang can track the sub-object.
   The ParameterBlock's own descriptor set is a separate entity at a different set index.

---

## Merian Architecture

### Overview

```
SlangProgram --> SlangProgramEntryPoint --> ShaderObject --> ShaderCursor
     |                   |                       |
     |            (pipeline layout,        (descriptors,
     |             set index cache,         uniform data,
     |             nested PB tree)          sub-objects)
     |
SlangObjectLayout --- (cached per type: descriptor set layout,
                        binding info, sub-object range map)
```

### Class Responsibilities

**`SlangProgram`** -- Wraps a compiled Slang program. Provides access to program
reflection (`ProgramLayout*`) and SPIR-V binary. Keeps the Slang session alive so all
reflection pointers remain valid.

**`SlangProgramEntryPoint`** -- Represents one entry point. Owns the pipeline layout
cache and the `ParameterBlockInfo` cache (maps parameter names to `SlangObjectLayout` +
descriptor set index + nested ParameterBlock tree). The `bind()` method is the main
entry point for binding shader objects at dispatch time.

**`SlangObjectLayout`** -- Cached reflection data for one Slang type. Created once per
type and reused. Stores:
- `DescriptorSetLayout` built from reflection
- Uniform data size (0 for resource-only types)
- `binding_info_cache`: O(1) lookup from binding range index to `BindingInfo{vk_binding, type, count}`
- `binding_range_to_subobject_range`: maps binding ranges to sub-object ranges
- `subobject_ranges`: pre-computed sub-object info with element layouts for CB and PB
  types (other types like RWStructuredBuffer are skipped to prevent infinite recursion)

**`ShaderObject`** -- The runtime parameter container. Stores:
- `DescriptorStorage` for caching descriptor writes (replayed to new sets each frame)
- `ordinary_data_staging` + `ordinary_data_buffer` for uniform data
- `subobjects` vector indexed by sub-object range
- `registered_sets` (weak pointers) for incremental write propagation
- `pb_bindings_` for CB sub-objects: tracks all owning PBs so nested CB descriptors
  are written at update time, not bind time

**`ShaderCursor`** -- Lightweight `(ShaderObject*, TypeLayout*, ShaderOffset)` tuple for
navigating the parameter space. Auto-creates ConstantBuffer and ParameterBlock
sub-objects on navigation. Supports `cursor["field"]["nested_field"] = value` syntax.
Navigating to a CB field auto-dereferences: `cursor["cb"]` returns a cursor into the
element type T, not the ConstantBuffer wrapper.

**`ShaderObjectAllocator`** -- Abstract allocator for descriptor sets and buffers.
`FrameCachingShaderObjectAllocator` caches sets per `(object, frame_index)` to avoid
re-allocation every frame. Call `set_iteration(frame_index)` before binding to select
the current frame's cache slot.

### Descriptor Write Model: Update-Time, Not Bind-Time

All descriptor writes happen eagerly when the user sets a value (via cursor or
`set_subobject`), NOT at bind time. This means a steady-state frame where nothing
changed requires zero descriptor writes.

```
Update time (cursor assignment):                   Bind time (dispatch):
================================                    ====================

cursor["tex"] = texture;                            bind_as_parameter_block()
  |                                                   |
  +-> descriptors->queue_write(...)  (storage)        +-> allocate set
  +-> for_each_registered_set(...)   (live sets)      +-> if new: replay from storage
                                                      +-> upload dirty staging data
set_subobject(sor, cb_obj):                          +-> set->update() + bind()
  |
  +-> write CB UBO to owning PB's storage/sets        (zero writes if nothing changed
  +-> set cb's pb_bindings_ for nested propagation     and set was already tracked)
```

For ConstantBuffer sub-objects, `set_subobject` writes the CB's UBO descriptor to
the owning ParameterBlock's descriptor storage and all registered sets. The CB tracks
its owning PBs via `pb_bindings_` (a vector, since a CB can be shared across multiple
PBs). Nested CBs (CB inside CB) propagate writes through the `pb_bindings_` chain to
the ultimate owning PB.

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
entry_point->bind("params", params, allocator, cmd, pipeline);
cmd->dispatch(extent, 16, 16);
```

The `bind()` call triggers this chain:

```
entry_point->bind("params", params, allocator, cmd, pipeline)
  |
  +- params->bind_as_parameter_block(cmd, pipeline, set=0)
  |    +- allocator->allocate(params) -> descriptor set for set 0
  |    +- if new set: descriptors->replay_to(set)
  |    +- if dirty: upload params' ordinary data (roughness)
  |    +- upload_constant_buffer_tree(light)     <- staging only, no descriptor writes
  |    +- set->update() + set->bind(cmd, pipeline, 0)
  |
  +- bind_nested_pbs(params, nested_pb_infos)
       |
       +- nested->bind_as_parameter_block(cmd, pipeline, set=1)
       |    +- if dirty: upload nested's ordinary data (weight)
       |    +- set->bind(cmd, pipeline, 1)
       |
       +- bind_nested_pbs(nested, nested_pb_infos.children)
            |
            +- deep->bind_as_parameter_block(cmd, pipeline, set=2)
                 +- if dirty: upload deep's ordinary data (value)
                 +- set->bind(cmd, pipeline, 2)
```

**Key design point:** CB descriptor writes (light's UBO at binding 1, etc.) were already
written to the PB's descriptor storage by `set_subobject` when the cursor navigated to
`cursor["light"]`. At bind time, only staging data uploads (dirty-guarded) and
descriptor set binding occur. In a steady-state frame, `bind_as_parameter_block` is
essentially: get cached set, bind it.

### Descriptor Set Index Assignment

Vulkan descriptor set indices are assigned sequentially in DFS order of ParameterBlock
nesting, **skipping PBs whose element type has no bindings**. This matches Slang's
SPIR-V output. The assignment happens in `get_pipeline_layout()` via
`collect_nested_pb_layouts()`:

```cpp
// DFS walk: each non-empty ParameterBlock gets next_set++
//   ParameterBlock<RootParams>     -> set 0  (has bindings)
//     ParameterBlock<NestedParams> -> set 1  (has bindings)
//       ParameterBlock<DeepParams> -> set 2  (has bindings)

// Empty PBs are skipped:
//   ParameterBlock<Wrapper>        -> NO SET (only contains a nested PB, no own bindings)
//     ParameterBlock<Inner>        -> set 0  (has bindings)
```

The indices are cached in a tree of `NestedPBInfo{subobject_range_index, set_index, children}`.
This tree mirrors the ParameterBlock nesting structure and is used by `bind_nested_pbs`
to know which set index to pass to each nested ParameterBlock during binding.
PBs with `set_index == NO_DESCRIPTOR_SET` (UINT32_MAX) are skipped at bind time.

### Auto-Creation of Sub-Objects

When `ShaderCursor::field()` navigates to a ConstantBuffer or ParameterBlock field, it
automatically creates the sub-object if it doesn't exist:

```cpp
// In ShaderCursor::field(uint32_t index):
if (field_kind == ConstantBuffer || field_kind == ParameterBlock) {
    auto& sub = base_object->subobjects[sor];
    if (!sub) {
        sub = make_shared<ShaderObject>(context, element_layout, allocator);
        // set_subobject handles CB descriptor writes to the owning PB
        base_object->set_subobject(sor, sub);
    }
    // Return cursor into the sub-object's element type (auto-dereference)
    return sub->get_cursor();
}
```

This means `cursor["nested"]["deep"]["value"] = 42.0f` transparently creates both
the `nested` and `deep` ShaderObjects on first access. The user never needs to manually
create sub-objects.

### Write Propagation

When a value-bound ShaderObject is written to, the CPU staging buffer is updated and
marked dirty. The actual GPU upload happens at bind time (only if dirty).

Resource writes (textures, buffers) go through `DescriptorStorage` which caches all
writes and can replay them to newly allocated descriptor sets. The write is also
immediately applied to all registered (live) descriptor sets. This means if the
allocator gives back an existing set, no replay is needed.

CB descriptor writes follow a different path: `set_subobject` writes the CB's UBO
descriptor directly to the owning PB's storage and registered sets. For nested CBs
(CB inside CB), the write propagates through the `pb_bindings_` chain. Since a CB can
be shared across multiple PBs, `pb_bindings_` is a vector of `{weak_ptr<PB>, binding}`
pairs. Expired entries are pruned lazily.

### Descriptor Set Layout Construction

`create_descriptor_set_layout_from_slang_type_layout()` in `slang_utils.cpp` builds a
`VkDescriptorSetLayout` from a Slang type layout. Key details:

- If the type has uniform data (`getSize(UNIFORM) > 0`), an implicit
  `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` is added at binding 0.
- Descriptor ranges from set 0 are iterated to add further bindings.
- A `std::map<uint32_t, VkDescriptorSetLayoutBinding>` is used for deduplication:
  CB element types (from `getElementVarLayout()->getTypeLayout()`) may report a
  ConstantBuffer descriptor range at binding 0 that overlaps with the implicit UBO.
  The manual UBO takes priority.

### Resource-Only ParameterBlocks

A ParameterBlock whose element type has no value fields (e.g., a struct containing only
textures) is valid and works correctly. The uniform size will be 0, so no uniform buffer
is allocated at binding 0. Only the resource descriptors are written.

```slang
struct TextureOnly {
    Texture2D<float4> tex;  // resource only, no uniform data
};

struct MyParams {
    ParameterBlock<TextureOnly> tex_only;  // valid: no UBO needed
};
```

### Debug Formatting

All key classes have `format_*` methods that print reflection information:

```cpp
SPDLOG_INFO("{}", program->format_reflection());
SPDLOG_INFO("{}", entry_point->format_reflection(context));
SPDLOG_INFO("{}", params->format_debug());
SPDLOG_INFO("{}", cursor.format_debug());
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
| **Descriptor management** | Deferred: all writes happen at bind time via `BindingDataBuilder` | Eager: writes cached in `DescriptorStorage`, replayed to new sets |
| **Uniform data** | CPU byte buffer (`m_data`), bulk-copied to pooled buffer at bind time | CPU staging buffer, uploaded via `StagingMemoryManager` per-buffer |
| **Resource storage** | Flat `m_slots` array of `ResourceSlot` tagged unions | `DescriptorStorage` (mirrors `VkDescriptorSet` writes) |
| **Sub-object storage** | Flat `m_objects` array indexed by sub-object range | `subobjects` vector indexed by sub-object range |
| **Pipeline layout** | `RootShaderObjectLayout` owns `VkPipelineLayout` + all set layouts | `SlangProgramEntryPoint` builds and caches pipeline layout |
| **Push constants** | Built-in: entry point data routed to push constants | Not used (all data goes through descriptor sets) |
| **Specialization** | Built-in: existential/interface types with RTTI + witness tables | Not yet supported |
| **Caching** | Version-based (`m_uid` + `m_version`) with `finalize()` | Frame-based via `FrameCachingShaderObjectAllocator` |
| **Parameter navigation** | `ShaderCursor` (library-provided, stateless) | `ShaderCursor` (similar, but auto-creates sub-objects) |
| **CB descriptor writes** | At bind time via `BindingDataBuilder` tree walk | At update time via `set_subobject` with `pb_bindings_` propagation |

### Key Design Decisions

**1. Eager vs. Deferred Descriptor Writes**

slang-rhi stores raw resource references and writes all descriptors at bind time.
Merian writes descriptors eagerly (cached in `DescriptorStorage`) and replays them
to new descriptor sets. This means descriptor writes happen once and are replayed,
rather than rebuilt every frame. In a steady-state frame, zero descriptor writes occur.

```cpp
// slang-rhi: resource stored as slot, written at bind time
shaderObject->setBinding(offset, texture);  // stores in m_slots[]
// ... later, BindingDataBuilder walks m_slots and calls vkUpdateDescriptorSets

// Merian: descriptor written immediately, cached for replay
cursor["tex"] = texture;  // calls vkUpdateDescriptorSets now, cached in DescriptorStorage
// ... later, new sets get replayed from cache: descriptors->replay_to(*new_set)
```

**2. Uniform Data Upload**

slang-rhi allocates a constant buffer from a pool at bind time and bulk-copies the
entire `m_data` byte buffer. Merian allocates a persistent buffer per ShaderObject and
uploads only when dirty via `StagingMemoryManager`.

**3. Sub-Object Auto-Creation**

slang-rhi requires explicit `setObject()` calls to create and attach sub-objects.
Merian's cursor auto-creates sub-objects when navigating to ConstantBuffer or
ParameterBlock fields:

```cpp
// slang-rhi: explicit creation and binding
auto lightObj = device->createShaderObject(lightLayout);
lightCursor["intensity"].setData(1.5f);
paramsCursor["light"].setObject(lightObj);

// Merian: transparent auto-creation
cursor["light"]["intensity"] = 1.5f;  // auto-creates light ShaderObject
```

**4. Descriptor Set Index Assignment**

slang-rhi's `RootShaderObjectLayout` computes set indices during layout construction
using `findOrAddDescriptorSet(space)` to map Slang register spaces to sequential
Vulkan descriptor set indices. Merian uses a simpler DFS walk that skips empty PB
layouts, matching Slang's SPIR-V output where types with no bindings don't get
descriptor sets.

**5. CB Binding Offset Computation**

Both slang-rhi and Merian use the same Slang reflection APIs for computing nested CB
bindings: `getContainerVarLayout()` gives the UBO binding offset, and
`getElementVarLayout()` gives where the element's content starts. slang-rhi accumulates
these through a `SimpleBindingOffset` struct; Merian computes them in
`compute_cb_binding_deltas()` and stores the result in `pb_bindings_`.

### What slang-rhi Has That Merian Doesn't (Yet)

- **Existential/interface type** support with RTTI headers and witness tables
- **Specialization** caching for generic shader objects
- **Push constant** routing for entry-point parameters
- **Finalization** (`finalize()`) for immutable, reusable shader objects
- **Global parameter** support (module-scope bindings)
- **Cross-backend** support (D3D12, Metal, CUDA, WGPU)
