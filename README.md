# Merian 🎨

~ _A Vulkan development framework._ ~


Merian is split into multiple components:

 - [`merian`](https://github.com/LDAP/merian/tree/main/include/merian): Provides core abstractions and utilities (Vulkan context, memory allocation, configuration, IO, ...).
 - [`merian-nodes`](https://github.com/LDAP/merian/tree/main/include/merian-nodes): Implements an extensible Vulkan processing graph. Already implemented nodes can be found [here](https://github.com/LDAP/merian/tree/main/include/merian-nodes/nodes).
 - [`merian-shaders`](https://github.com/LDAP/merian/tree/main/include/merian-shaders): Collection of reusable shader code.

## Examples

- [merian-quake](https://github.com/LDAP/merian-quake): A path-tracer for the original Quake game.
- [merian-shadertoy](https://github.com/LDAP/merian-shadertoy): A limited Vulkan implementation for Shadertoys with hot reloading.
- [merian-hdr-viewer](https://github.com/LDAP/merian-hdr-viewer): A simple HDR viewer with various exposure and tone-mapping controls.
- [merian-example-sum](https://github.com/LDAP/merian-example-sum): Example on how to compute a sum on the GPU.

Merian aims for compatibility with Windows, Linux as well as all major GPU vendors.

## Getting started

```c++
int main() {
    // "merian-resources" initializes a memory allocator, resource allocator and staging manager.
    // The core "merian" extension is always loaded automatically.
    merian::ContextHandle context = merian::Context::create({
        .context_extensions = {"merian-resources", "merian-debug-utils"},
        .application_name = "my-app",
    });

    auto alloc = context->get_context_extension<merian::ExtensionResources>()->resource_allocator();

    // allocating, rendering, ...

    // merian cleans up everything for you
}
```

To enable Vulkan extensions (e.g. ray tracing acceleration structures):

```c++
merian::ContextHandle context = merian::Context::create({
    .device_extensions = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME},
    .context_extensions = {"merian-resources"},
});
```

To enable Vulkan features (e.g. ray tracing):

```c++
merian::VulkanFeatures features;
features.set_feature("accelerationStructure", true);
features.set_feature("rayQuery", true);

merian::ContextHandle context = merian::Context::create({
    .features = features,
    .context_extensions = {"merian-resources", "merian-glfw"},
});
```

## Include Merian into your Project

This library uses the [Meson Build System](https://mesonbuild.com/) and declares a dependency for it:

``` py
# in your meson.build

merian = dependency('merian', version: ['>=2.0.0'], fallback: ['merian', 'merian_dep'])
# or if you want to use the shader compiler from merian
merian_subp = subproject('merian', version: ['>=2.0.0'])
merian = merian_subp.get_variable('merian_dep')
shader_generator = merian_subp.get_variable('shader_generator')

sources = ['app.cpp']
sources += shader_generator.process('shader0.comp', 'shader1.comp',...)
# or to prevent conflicts:
sources += shader_generator.process(
    'shader0.comp', 'shader1.comp',...
    extra_args: ['--prefix', 'my_app_c_prefix'],
    preserve_path_from: meson.source_root(),
)

exe = executable(
    'my-app',
    sources,
    dependencies: [
        merian,
    ],
    # ...
)


```

To allow meson to find Merian, either clone this repo into the `subprojects` folder or add a file `subprojects/merian.wrap` with

```ini
[wrap-git]
directory = merian

url = https://github.com/LDAP/merian
revision = 2.0.0
depth = 1
clone-recursive = true

[provide]
merian = merian_dep
```

## Documentation

Documentation is in the `docs` subdirectory of this repository.

Nodes are documented in their [respective subfolder](https://github.com/LDAP/merian/tree/main/include/merian-nodes/nodes).

## Usage

Merian is similar to the `vulkan_raii.hpp` layer for `vulkan.hpp`. Most objects follow the RAII principle. The `merian` namespace provides shareable handles for most Vulkan types (e.g. `merian::ImageHandle` for `vk::Image`) and objects are automatically destroyed if their reference count becomes 0.

The `Context` class initializes and destroys a Vulkan device and holds core objects (PhysicalDevice, Device, Queues, ...).

Create a Context using `Context::create(ContextCreateInfo)`. The core `"merian"` extension is always loaded automatically. Additional extensions are loaded by name via `ContextCreateInfo::context_extensions`. Vulkan features are requested via `ContextCreateInfo::features` using a `VulkanFeatures` object.

Make sure your program ends with `[INFO] [context.cpp:XX] context destroyed`.

Note that the Vulkan dynamic dispatch loader must be used. The default dispatcher is initialized in `Context`. The merian build system should already ensure that.

```c++
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
```
