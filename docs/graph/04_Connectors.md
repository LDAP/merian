## Connectors

Connectors define the typed I/O of a node. The graph uses them to allocate resources, insert barriers, and wire descriptor set bindings automatically.

Names are **not** part of the connector factory method — they are supplied in `InputConnectorDescriptor` / `OutputConnectorDescriptor` when returned from `describe_inputs()` / `describe_outputs()`.

### Delay system

Input connectors accept a `delay` parameter. A delay of N means the node receives data from N iterations ago. The graph allocates (delay + 1) resource copies and cycles through them each iteration. Delay 0 (default) means the current frame's output from the upstream node.

---

## Input connectors

### VkBufferIn

Storage buffer input. A descriptor binding is only created when stage flags are specified (which the factory methods handle automatically).

```cpp
// Factory methods (all return VkBufferInHandle):
VkBufferIn::compute_read(delay=0, optional=false, usage=eStorageBuffer)
VkBufferIn::fragment_read(delay=0, optional=false, usage=eStorageBuffer)
VkBufferIn::acceleration_structure_read(delay=0, optional=false)
VkBufferIn::transfer_src(delay=0, optional=false)

// In process(): io[buffer_in] -> const BufferArrayResource&
//   .buffer  - the BufferHandle for the current frame
```

### VkSampledImageIn

Sampled image input. Transitions layout to `eShaderReadOnlyOptimal` and creates a combined image sampler descriptor binding.

```cpp
VkSampledImageIn::compute_read(delay=0, optional=false, sampler=std::nullopt)
VkSampledImageIn::fragment_read(delay=0, optional=false)

// In process(): io[image_in] -> const ImageArrayResource&
//   .image      - the ImageHandle
//   .image_view - the ImageViewHandle
```

### VkImageInStorage

Storage image input (read or read/write). Keeps layout as `eGeneral`.

```cpp
VkImageInStorage::compute_read(delay=0, optional=false)
// ... other variants

// In process(): io[image_in] -> const ImageArrayResource&
```

### VkTlasIn

Top-level acceleration structure input for ray tracing.

```cpp
VkTlasIn::compute_read()
VkTlasIn::fragment_read()

// In process(): io[tlas_in] -> const TlasResource&
//   .tlas - the AccelerationStructureHandle
```

### PtrIn / PtrOut

Host-pointer connectors for passing CPU-side data through the graph without GPU resources. Useful for configuration structs or scene data.

```cpp
// PtrIn<T> / PtrOut<T> — typed host pointer
// In process(): io[ptr_in] -> T* (may be nullptr if not connected)
```

### AnyIn / AnyOut

Generic connectors that accept any resource type. Use when the connector type is determined at runtime or for pass-through nodes.

---

## Output connectors

### ManagedVkImageOut

The graph allocates and manages the image. The node writes to it each frame.

```cpp
// Factory methods (all return ManagedVkImageOutHandle):
ManagedVkImageOut::compute_write(format, extent, persistent=false)
ManagedVkImageOut::compute_fragment_write(format, extent, persistent=false)
ManagedVkImageOut::fragment_write(format, extent, persistent=false)
ManagedVkImageOut::color_attachment(format, extent)

// persistent=true: resource is not cleared between frames (for accumulators).

// In process(): io[image_out] -> const ImageArrayResource&
//   .image      - the ImageHandle
//   .image_view - the ImageViewHandle
```

### ManagedVkBufferOut

The graph allocates and manages the buffer.

```cpp
ManagedVkBufferOut::compute_write(size, usage)

// In process(): io[buffer_out] -> const BufferArrayResource&
//   .buffer - the BufferHandle
```

### VkTlasOut

Top-level acceleration structure output. The node fills in the TLAS handle.

```cpp
VkTlasOut::create(name)

// In process(): io[tlas_out] -> TlasResource& (writable)
//   Set .tlas to the AccelerationStructureHandle built this frame.
```

---

## Descriptor bindings

Each connector that requires GPU access gets an automatically assigned descriptor binding. In `process()`, the binding index can be queried if needed:

```cpp
uint32_t b = io.get_binding(my_input_connector);
uint32_t b = io.get_binding(my_output_connector);
```

The descriptor set passed to `process()` already contains up-to-date bindings for all connectors — there is no need to manually update them.
