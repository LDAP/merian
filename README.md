# Merian 🎨

~ _A *Vulkan 1.3* prototyping framework._ ~


Merian is split into two libraries:

 - `merian`: Provides core abstractions and utilities (Vulkan context, memory allocation, configuration, IO, ...).
 - `merian-nodes`: Implements an extensible Vulkan processing graph.


## Examples

- [merian-quake](https://github.com/LDAP/merian-quake-rt): A path-tracer for the original Quake game. (coming soon)
- [merian-shadertoy](https://github.com/LDAP/merian-shadertoy): A limited Vulkan executor for Shadertoys with hot reloading
- [merian-hdr-viewer](https://github.com/LDAP/merian-hdr-viewer): A simple HDR viewer with various exposure and tone-mapping controls.

## Getting started

```c++
int main() {
    auto debug_utils = std::make_shared<merian::ExtensionVkDebugUtils>(false);
    auto resources = std::make_shared<merian::ExtensionResources>();

    merian::SharedContext context = merian::Context::make_context({debug_utils, resources}, "merian");
    auto alloc = resources->resource_allocator();

    // allocating, rendering,...

    // merian cleans up everything for you
}    
```

## Include Merian into your Project

This library uses the [Meson Build System](https://mesonbuild.com/) and declares a dependency for it:

``` py
# in your meson.build

merian = dependency('merian', version: ['>=1.0.0'], fallback: ['merian', 'merian_dep'])
exe = executable(
    'my-app',
    dependencies: [
        merian,
    ],
    // ...
)

```

If Merian is not installed, either clone Merian into the subprojects folder or add a file `subprojects/merian.wrap` with

```ini
[wrap-git]
directory = merian

url = https://github.com/LDAP/merian
revision = 1.0.0
depth = 1
clone-recursive = true

[provide]
merian = merian_dep
```

## Documentation

Documentation is in the `docs` subdirectory of this repository.

## Usage

Merian is similar to the `vulkan_raii.hpp` layer for `vulkan.hpp`. And most objects follow the RAII principle. In most cases you want to wrap them in smart pointers.
Merian provides aliases for these types (e.g. `merian::ImageHandle` for `std::shared_ptr<merian::Image>`).

The `Context` class initializes and destroys a Vulkan device and holds core objects (PhysicalDevice, Device, Queues, ...).

Create a Context using the `Context::make_context(..)` method.

For common Vulkan objects a wrapper is provided that holds a shared pointer on its parent.
This is to ensure that dependent objects (Command Buffer → Pool → Device) are destructed in the right order automagically.
You should never need to call `destroy()` or `free()` manually.
Keep in mind to hold a reference to the shared pointers to prevent frequent object construction and destruction.
Whenever you create objects by yourself you should consider using `make_shared<Class>(..)`.
If a `std::bad_weak_ptr` is thrown you should have used `make_shared<>(...)` instead of creating the object directly.

Make sure your program ends with `[INFO] [context.cpp:XX] context destroyed`.

Note that the Vulkan dynamic dispatch loader must be used. The default dispatcher is initialized in `Context`.

```c++
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
```

### Lifetime of objects

Design decisions:

- Functions that don't impact an object's lifetime (i.e. the object remains valid for the duration of the function) should take a plain reference or pointer, e.g. `int foo(bar& b)`.
- Functions that consume an object (i.e. are the final users of a given object) should take a `unique_ptr` by value, e.g. `int foo(unique_ptr<bar> b)`. Callers should std::move the value into the function.
- Functions that extend the lifetime of an object should take a `shared_ptr` by value, e.g. `int foo(shared_ptr<bar> b)`. The usual advice to avoid circular references applies. This does the heavy lifting of destroying objects in the right order.
- For Vulkan objects with lifetime a wrapper is provided that destroys the Vulkan object in its destructor.
  These object should derive from `std::enable_shared_from_this`.


## Licenses

- This project uses parts from https://github.com/nvpro-samples/nvpro_core, which is licensed under [Apache License Version 2.0, January 2004](https://github.com/nvpro-samples/nvpro_core/blob/master/LICENSE). Copyright notice:
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
- Replace buffer suballocator with VMAVirtualAllocator
- Graph: Allocator: e.g. free memory for "later" nodes?, Parallel cmd recording.
