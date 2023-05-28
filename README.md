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
The Context can be extended by Extensions. The extensions can hook into the context creation process and enable Vulkan instance/device layers and extensions as well as features.

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
