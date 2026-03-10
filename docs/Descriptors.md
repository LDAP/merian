## Descriptors

Merian provides a builder for `DescriptorSetLayouts` as well as a `DescriptorSetUpdate` utility class which eases binding descriptors to resources.

- `DescriptorSetLayoutBuilder`: A builder for `DescriptorSetLayout`.
- `DescriptorSetLayout`: Wraps a descriptor set layout and destroys it in its destructor. Makes it easy to generate a pool from this layout.
- `DescriptorPool`: Wraps a descriptor pool and destroys it in its destructor. Makes it easy to generate sets from this pool.
- `DescriptorSet`: Wraps a descriptor set and destroys it in its destructor.

### Example

```c++
auto desc_layout = merian::DescriptorSetLayoutBuilder()
    .add_binding_storage_buffer()           // binding 0: result buffer
    .add_binding_acceleration_structure()   // binding 1: top-level AS
    .add_binding_storage_buffer()           // binding 2: vertex buffer
    .add_binding_storage_buffer()           // binding 3: index buffer
    .build_layout(context);

// DescriptorPool::create() is the public factory. DescriptorSet is allocated from the pool.
// (DescriptorSet constructor is private — use desc_pool->allocate())
auto desc_pool = merian::DescriptorPool::create(desc_layout);
auto desc_set  = desc_pool->allocate(desc_layout);

desc_set->queue_descriptor_write_buffer(0, result_buffer)
         .queue_descriptor_write_acceleration_structure(1, tlas)
         .queue_descriptor_write_buffer(2, vertex_buffer)
         .queue_descriptor_write_buffer(3, index_buffer);
desc_set->update();
```

### Push descriptors

For pipelines that use push descriptors (the graph uses push descriptors internally), resources can be pushed directly from the command buffer without allocating a descriptor pool or set:

```cpp
// Convenience overloads accept BufferHandle, TextureHandle, ImageViewHandle, ...
cmd->push_descriptor_set(pipeline, set_index, buffer, texture, image_view);
```
