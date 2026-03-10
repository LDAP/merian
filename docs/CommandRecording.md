## Command Recording

Merian provides several abstractions and helpers for command recording.

- `Queue`: Holds a queue together with its mutex. Provides convenience methods to submit using the mutex.
- `CommandPool`: Wraps a command pool that is automatically destroyed when the object is destroyed. Call `create_and_begin()` to allocate and immediately begin a command buffer.
- `RingCommandPool`: Holds a command pool for each frame in flight (see Synchronization).

### CommandBuffer

`CommandBufferHandle` wraps a `vk::CommandBuffer` and keeps referenced resources alive until the pool is reset.

**Lifecycle:**
```cpp
cmd->begin();
// ... record ...
cmd->end();
pool->reset();  // releases all kept resources
```

**Pipeline binding:**
```cpp
cmd->bind(pipeline);
cmd->dispatch(group_count_x, group_count_y, group_count_z);
cmd->dispatch(extent, local_size_x, local_size_y, local_size_z);  // computes ceil(extent/local)
```

**Descriptor sets:**
```cpp
cmd->bind_descriptor_set(pipeline, 0, desc_set);
cmd->bind_descriptor_set(pipeline, first_set, set_a, set_b);  // variadic
```

**Push descriptors (convenience overloads):**
```cpp
// Accepts BufferHandle, TextureHandle, ImageViewHandle, AccelerationStructureHandle, ...
cmd->push_descriptor_set(pipeline, set_index, buffer, texture, image_view);
```

**Push constants:**
```cpp
cmd->push_constant(pipeline, my_struct);          // id=0
cmd->push_constant(pipeline, my_struct, id);      // explicit constant id
```

**Buffer operations:**
```cpp
cmd->copy(src_buffer, dst_buffer, regions);
cmd->fill(buffer, value = 0);                     // fill entire buffer with uint32
cmd->update(dst_buffer, offset, data_array);      // inline data update (<= 64 KiB)
```

**Image operations:**
```cpp
cmd->copy(src_image, src_layout, dst_image, dst_layout, regions);
cmd->blit(src, src_layout, dst, dst_layout, regions, filter);
cmd->clear(image, layout, color, subresource_ranges);
```

**Barriers (Vulkan synchronization2, preferred):**
```cpp
// Single image — layout tracked automatically on the Image object:
cmd->barrier(image->barrier2(vk::ImageLayout::eGeneral));

// Single buffer:
cmd->barrier(buffer->buffer_barrier2(src_stage, dst_stage, src_access, dst_access));

// Batch:
cmd->barrier(dep_info);
cmd->barrier(image_barriers_array);
cmd->barrier(buffer_barriers_array);
```

**Barriers (Vulkan 1.2 style):**
```cpp
cmd->barrier(src_stage, dst_stage, memory_barriers);
cmd->barrier(src_stage, dst_stage, buffer_barriers);
cmd->barrier(src_stage, dst_stage, image_barriers);
```

**Render pass:**
```cpp
cmd->begin_render_pass(framebuffer, render_area, clear_values);
// ... draw calls ...
cmd->end_render_pass();
```

**Ray tracing:**
```cpp
cmd->copy_acceleration_structure(src, dst, mode);
```

**Query pools:**
```cpp
cmd->reset(query_pool, first_query, count);
cmd->write_timestamp(query_pool, query, pipeline_stage);
```

### One-shot submit example

The simplest one-shot pattern uses the `submit_wait` lambda overload (creates, records, submits, and waits internally):

```cpp
context->get_queue_GCT()->submit_wait([&](const merian::CommandBufferHandle& cmd) {
    // ... record work ...
});
```

`create_and_begin()` is on `CachingCommandPool`, not on the base `CommandPool`. For explicit control:

```cpp
auto pool = std::make_shared<merian::CachingCommandPool>(context->get_cmd_pool_GCT());
auto cmd  = pool->create_and_begin();

// ... record work ...

cmd->end();
context->get_queue_GCT()->submit_wait({cmd});
pool->reset();
```
