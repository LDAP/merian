# Merian Vulkan Framework

A framework for quick Vulkan prototyping.

## Usage

The Vulkan dynamic dispatch loader must be used.

```c++
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
```

The default dispatcher is initialized in `Context`.


### Context
The `Context` class initializes and destroys a Vulkan device and holds core objects (PhysicalDevice, Device, Queues, ...).

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

    merian::Context app(extensions, "My beautiful app");

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
- `ExtensionVkV12`: Enables Vulkan 1.2 features.

Ray tracing:
- `ExtensionVkRayTracingPipeline`: Enables ray tracing pipeline extensions and provides access to common functions.
- `ExtensionVkRayQuery`: Enables ray query extensions and provides access to common functions.
- `ExtensionVkAccelerationStructure`: Build and manage ray tracing acceleration structures extensions and provides access to common functions.


### Helpers

- `SamplerPool`: Reuse samplers with identical configuration (keeps reference count).
- `MemoryAllocator`: Interface for memory allocators (called MemoryAllocator in NVPro Core). Memory requirements are described by `MemAllocateInfo` and memory is referenced by `MemHandle`. 
- `MemoryAllocatorVMA`: A implementation of `MemoryAllocator` using the [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) (VMA).
- `ResourceAllocator`: Uses a `MemoryAllocator` to create and destroy resources.
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
        //...

        frame++;

        // wait until we can use the new cycle 
        // (very rare if we use the fence at then end once per-frame)
        ringFences.setCycleAndWait(frame);

        // update cycle state, allows recycling of old resources
        auto& cycle = ringPool.setCycle( frame );

        VkCommandBuffer cmd = cycle.createCommandBuffer(...);
        
        //... do stuff / submit etc...
        VkFence fence = ringFences.getFence();

        // use this fence in the submit
        queue_container.submit(cmd, fence);

        //...
    }
    ```

- `Camera`: Helper class to calculate view and projection matrices.
- `CameraAnimator`: Helper class to smooth camera motion.
- `CameraController`: Helper class to control a camera with high level commands.
- `Renderdoc`: Helper class to enable start frame capturing for RenderDoc.
- `FileLoader`: Helper class to find and load files from search paths.

## Building

```bash
meson setup build
# or to compile with debug flags
meson setup build --buildtype=debug

meson compile -C build
```

## Licenses

- Parts of this code are adapted from https://github.com/nvpro-samples/nvpro_core, which is licensed under [Apache License Version 2.0, January 2004](https://github.com/nvpro-samples/nvpro_core/blob/master/LICENSE). Copyright notice:
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

- Replace lots of manual `destroy()` with shared_ptrs and weak shared_ptrs. Especially in resource management, extensions, descriptorsets and pools. (When child objects have pointers to their parent they are destroyed before the parent is destroyed.) Start the tree at Context, this allows the childen to access the context to destroy itself?
- Extract Swapchain from ExtensionGLFW
- AS Builder: Only record commands, dont wait.
