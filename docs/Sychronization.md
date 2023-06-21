## Synchronization

Merian provides several abstractions and utilities for synchronization:

>In real-time processing, the CPU typically generates commands 
>in advance to the GPU and send them in batches for execution.
>
>To avoid having the CPU to wait for the GPU'S completion and let it "race ahead"
>we make use of double, or tripple-buffering techniques, where we cycle through
>a pool of resources every frame. We know that those resources are currently 
>not in use by the GPU and can therefore manipulate them directly.
>
>Especially in Vulkan it is the developer's responsibility to avoid such
>access of resources that are in-flight.
>
>The "Ring" classes cycle through a pool of resources. The default value
>is set to allow two frames in-flight, assuming one fence is used per-frame.
>
>_From https://github.com/nvpro-samples/nvpro_core/blob/master/nvvk/commands_vk.hpp_

- `Event`: Wraps a Vulkan event, and destroys it in its destructor
- `RingFences`: Provide a fence for each frame in flight, see example below.

### Example

    ```c++
    int main() {
        auto ring_cmd_pool =
            make_shared<merian::RingCommandPool<>>(context, context->queue_family_idx_GCT);
        auto ring_fences = make_shared<merian::RingFences<>>(context);
        auto [window, surface] = extGLFW.get();
        auto swap = make_shared<merian::Swapchain>(context, surface, queue);
        while (!glfwWindowShouldClose(*window)) {
            vk::Fence fence = ring_fences->wait_and_get_fence();
            auto cmd_pool = ring_cmd_pool->set_cycle();
            auto cmd = cmd_pool->create_and_begin();

            glfwPollEvents();
            auto aquire = swap->aquire_auto_resize(*window);
            assert(aquire.has_value());

            if (aquire->did_recreate) {
                swap->cmd_update_image_layouts(cmd);

                // recreate/update your stuff...
            }

            // do your usual stuff...

            cmd_pool->end_all();
            queue->submit(cmd_pool, fence, {aquire->signal_semaphore}, {aquire->wait_semaphore},
                          {vk::PipelineStageFlagBits::eColorAttachmentOutput});
            swap->present(*queue);
        }
    }
    ```
