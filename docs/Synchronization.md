## Synchronization

Merian provides several abstractions and utilities for synchronization:

>In real-time processing, the CPU typically generates commands
>in advance of the GPU and sends them in batches for execution.
>
>To avoid having the CPU wait for the GPU's completion and let it "race ahead",
>we make use of double or triple-buffering techniques, where we cycle through
>a pool of resources every frame. We know that those resources are currently
>not in use by the GPU and can therefore manipulate them directly.
>
>Especially in Vulkan it is the developer's responsibility to avoid
>access of resources that are in-flight.
>
>The "Ring" classes cycle through a pool of resources. The default value
>is set to allow two frames in-flight, assuming one fence is used per frame.

- `Event`: Wraps a Vulkan event and destroys it in its destructor.
- `RingFences<UserDataType>`: Provides a fence (and optional per-slot user data) for each frame in flight.
- `TimelineSemaphore`: A semaphore with a monotonically increasing counter. Can be waited on and signaled from the CPU.
- `BinarySemaphore`: The classic binary semaphore used mainly to synchronize multiple queue submits (e.g. swapchain acquire/present).

### RingFences

```c++
// Default 2 frames in flight. Optionally carry per-frame user data via the template parameter.
auto ring_fences = std::make_shared<merian::RingFences<MyFrameData>>(context);

while (true) {
    // Advance to the next slot, wait for its fence, reset it, and return RingData&.
    // RingData has .fence (vk::Fence) and .user_data (MyFrameData).
    merian::RingFences<MyFrameData>::RingData& ring = ring_fences->next_cycle_wait_reset_get();
    MyFrameData& frame = ring.user_data;
    vk::Fence fence    = ring.fence;

    // ... record and submit ...
    queue->submit(cmds, fence, ...);
}
```

### Queue submission

`Queue` wraps a `vk::Queue` with a mutex and provides convenience submit methods:

```c++
// Submit and wait for completion (convenience: creates + records + submits internally)
queue->submit_wait([&](const merian::CommandBufferHandle& cmd) {
    // record commands here
});

// Or submit already-recorded command buffers and wait:
queue->submit_wait(cmds);  // vk::ArrayProxy<CommandBufferHandle>

// Submit without blocking
queue->submit(submit_infos, fence);
queue->submit(cmds, fence, signal_semaphores, wait_semaphores, wait_stages);
```

### Swapchain

The `Swapchain` and `SwapchainManager` classes handle presentation:

```c++
// SwapchainManager wraps a Swapchain handle.
auto swapchain_manager = std::make_shared<merian::SwapchainManager>(swapchain);

// Acquire next image. Returns nullopt if minimized or acquire failed.
// Pass the window to auto-resize on extent change.
auto result = swapchain_manager->acquire(window);
if (!result) continue;

// result->image_view    - the current swapchain image view
// result->wait_semaphore   - signal: image is ready to write
// result->signal_semaphore - must be signaled when writing is done
// result->did_recreate  - true if swapchain was rebuilt this frame

// result->wait_semaphore is signaled by the presentation engine — wait on it before writing.
// result->signal_semaphore must be signaled by the app when writing is done.
queue->submit(cmds, fence,
              {*result->signal_semaphore},   // signal when writing is done
              {*result->wait_semaphore},      // wait: image is ready to write
              {vk::PipelineStageFlagBits::eColorAttachmentOutput});

// Present via the Swapchain (SwapchainManager::get_swapchain() returns the current one):
swapchain_manager->get_swapchain()->present(queue, result->index);
```
