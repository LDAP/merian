# Merian 🎨

~ _A *Vulkan* prototyping framework._ ~


## Building

```bash
meson setup build
# or to compile with debug flags
meson setup build --buildtype=debug

meson compile -C build
```


## Usage

The Vulkan dynamic dispatch loader must be used.

```c++
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
```

The default dispatcher is initialized in `Context`.

The `Context` class initializes and destroys a Vulkan device and holds core objects (PhysicalDevice, Device, Queues, ...).

Create a Context using the `Context::make_context(..)` method.

For common Vulkan objects a wrapper is provided that holds a shared pointer on its parent.
This is to ensure that dependent objects (Command Buffer -> Pool -> Device) are destructed in the right order automagically.
You should never need to call `destroy()` or `free()` manually. Keep in mind to keep a reference to the shared pointers to prevent frequent object construction and destruction. Whenever you create objects by yourself you should consider using `make_shared<Class>(..)` (see below).
If a `std::bad_weak_ptr` is thrown you should have used `make_shared<>(...)` instead of creating the object yourself.

Make sure your program ends with `[INFO] [context.cpp:XX] context destroyed`.

Example:
```c++
int main() {
    merian::ExtensionVkDebugUtils debugUtils;
    merian::ExtensionResources resources;
    merian::ExtensionVkMaintenance4 maintenance4;
    std::vector<merian::Extension*> extensions = {&debugUtils, &resources, &maintenance4};

    merian::SharedContext context = merian::Context::make_context(extensions, "My beautiful app");
    auto alloc = resources.resource_allocator();

    const uint32_t width = 800;
    const uint32_t height = 600;

    merian::BufferHandle result = alloc->createBuffer(
        width * height * 3 * sizeof(float),
        vk::BufferUsageFlagBits::eStorageBuffer,
        merian::HOST_ACCESS_RANDOM
    );

    // Cleans up automagically (first buffer then resource allocator then memory allocator then ... then context)
}
```

### Extensions
The Context can be extended by Extensions. The extensions can hook into the context creation process and enable Vulkan instance/device layers and extensions as well as features. Note: Extension can only be used after `Context` is initialized.

```c++

int main() {
    merian::ExtensionVkDebugUtils debugUtils;
    merian::ExtensionVkGLFW extGLFW;
    merian::ExtensionResources resources;
    std::vector<merian::Extension*> extensions = {
        &extGLFW,
        &debugUtils,
        &resources
    };

    merian::SharedContext context = merian::Context::make_context(extensions, "My beautiful app");

    // use extensions...
}
```

#### Naming conventions:

- `Extension*`: General name of an extension
- `ExtensionVk*`: Name of an extension that enables / requires / is associated with Vulkan extensions or features.

#### Current extensions:

Common:

- `ExtensionVkDebugUtils`: Enables Validation Layers.
- `ExtensionVkGLFW`: Initializes GLFW and creates window, surface and swapchain.
- `ExtensionResources`: Convenience extension to initialize a `ResourceAllocator`, `MemoryAllocator` and `SamplerPool` using `MemoryAllocatorVMA`.
- `ExtensionStopwatch`: Convenience extension to create timestamps on a CommandBuffer and retrieve the time.

Features:

- `ExtensionVkFloatAtomics`: Enables float atomics (`VK_EXT_shader_atomic_float`).
- `ExtensionVkMaintenance4`: Enables maintenance4 extension and feature

Ray tracing:

- `ExtensionVkRayTracingPipeline`: Enables ray tracing pipeline extensions and provides access to common functions.
- `ExtensionVkRayQuery`: Enables ray query extensions and provides access to common functions.
- `ExtensionVkAccelerationStructure`: Enables extensions to build ray tracing acceleration structures.

### Helpers

- `SamplerPool`: Reuse samplers with identical configuration (keeps reference count).
- `MemoryAllocator`: Interface for memory allocators (called MemoryAllocator in NVPro Core). Memory requirements are described by `MemAllocateInfo` and memory is referenced by `MemHandle`. 
- `MemoryAllocatorVMA`: A implementation of `MemoryAllocator` using the [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) (VMA).
- `ResourceAllocator`: Uses a `MemoryAllocator` to create and destroy resources.
  Example:
    ```c++
    auto mem_alloc = VMAMemoryAllocator::make_allocator(context);
    auto sampler_pool = std::make_shared<SamplerPool>(context);
    auto staging = std::make_shared<StagingMemoryManager>(context, mem_alloc);
    auto alloc = std::make_shared<ResourceAllocator>(context, mem_alloc, staging, sampler_pool);

    const uint32_t width = 800;
    const uint32_t height = 600;

    merian::BufferHandle result = alloc->createBuffer(
        width * height * 3 * sizeof(float),
        vk::BufferUsageFlagBits::eStorageBuffer,
        merian::HOST_ACCESS_RANDOM
    );

    // Render...

    auto data = result->get_memory()->map();
    stbi_write_hdr("out.hdr", width, height, 3, reinterpret_cast<float*>(data));
    result->get_memory()->unmap();
    ```

- `QueueContainer`: Holds a queue together with its mutex. Provides convenience methods to submit using the mutex.
- `Ring*`:
    In real-time processing, the CPU typically generates commands 
    in advance to the GPU and send them in batches for execution.

    To avoid having the CPU to wait for the GPU'S completion and let it "race ahead"
    we make use of double, or tripple-buffering techniques, where we cycle through
    a pool of resources every frame. We know that those resources are currently 
    not in use by the GPU and can therefore manipulate them directly.
  
    Especially in Vulkan it is the developer's responsibility to avoid such
    access of resources that are in-flight.

    The "Ring" classes cycle through a pool of resources. The default value
    is set to allow two frames in-flight, assuming one fence is used per-frame.

    >Cite from https://github.com/nvpro-samples/nvpro_core/blob/master/nvvk/commands_vk.hpp

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

- `Camera`: Helper class to calculate view and projection matrices.
- `CameraAnimator`: Helper class to smooth camera motion.
- `CameraController`: Helper class to control a camera with high level commands.
- `Renderdoc`: Helper class to enable start frame capturing for RenderDoc.
- `FileLoader`: Helper class to find and load files from search paths.


- `Descriptor*`: Create and manage Descriptor Sets, Pools and SetLayouts. Usage:
    ```c++
    auto builder = merian::DescriptorSetLayoutBuilder()
        .add_binding_storage_buffer()           // result
        .add_binding_acceleration_structure()   // top-level acceleration structure
        .add_binding_storage_buffer()           // vertex buffer
        .add_binding_storage_buffer();          // index buffer

    auto desc_layout = builder.build_layout(context);
    // Shared pointers are used -> allows automatic destruction in the right order.
    auto desc_pool = std::make_shared<merian::DescriptorPool>(desc_layout);
    auto desc_set = std::make_shared<merian::DescriptorSet>(desc_pool);

    merian::DescriptorSetUpdate(desc_set)
        .write_descriptor_buffer(0, result)
        .write_descriptor_acceleration_structure(1, {as})
        .write_descriptor_buffer(2, vertex_buffer)
        .write_descriptor_buffer(3, index_buffer)
        .update(context);
    ```

- `Pipeline*` / `SpecicalizationInfo*`
    ```c++
    auto shader = std::make_shared<merian::ShaderModule>(context, "raytrace.comp.glsl.spv", loader);
    auto pipeline_layout = merian::PipelineLayoutBuilder(context)
                               .add_descriptor_set_layout(desc_layout)
                               .build_pipeline_layout();
    auto spec_builder = merian::SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y); // contant ids 0 and 1
    auto spec_info = spec_builder.build();
    auto pipeline = merian::ComputePipeline(pipeline_layout, shader, spec_info);

    cmd = pool->create_and_begin();
    pipeline.bind(cmd);
    pipeline.bind_descriptor_set(cmd, desc_set);

    cmd.dispatch((uint32_t(width) + local_size_x - 1) / local_size_x,
                 (uint32_t(height) + local_size_y - 1) / local_size_y, 1);
    merian::cmd_barrier_compute_host(cmd);
    pool->end_all();
    queue->submit_wait(pool);
    pool->reset();
    ```

### Lifetime of objects

Design decisions:

- Functions that don't impact an object's lifetime (i.e. the object remains valid for the duration of the function) should take a plain reference or pointer, e.g. `int foo(bar& b)`.
- Functions that consume an object (i.e. are the final users of a given object) should take a `unique_ptr` by value, e.g. `int foo(unique_ptr<bar> b)`. Callers should std::move the value into the function.
- Functions that extend the lifetime of an object should take a `shared_ptr` by value, e.g. `int foo(shared_ptr<bar> b)`. The usual advice to avoid circular references applies. This does the heavy lifting of destroying objects in the right order.
- For Vulkan objects with lifetime a wrapper is provided that destroys the Vulkan object in its destructor.
  These object should derive from `std::enable_shared_from_this`.

## Licenses

- Parts of this code is adapted from https://github.com/nvpro-samples/nvpro_core, which is licensed under [Apache License Version 2.0, January 2004](https://github.com/nvpro-samples/nvpro_core/blob/master/LICENSE). Copyright notice:
    ```
    Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
    SPDX-License-Identifier: Apache-2.0
    ```


## ToDO

- Extract Swapchain from ExtensionGLFW
- AS Builder: Only record commands, dont wait.
- Replace buffer suballocator with VMAVirtualAllocator
