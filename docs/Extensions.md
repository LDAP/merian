## Extensions

The Context can be extended by Extensions.
The extensions can hook into the context creation process and enable Vulkan instance/device layers and extensions as well as features.
Note: Extension can only be used after `Context` is initialized.

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

### Naming conventions:

- `Extension*`: General name of an extension
- `ExtensionVk*`: Name of an extension that enables / requires / is associated with Vulkan extensions or features.

### Current extensions:

Common:

- `ExtensionVkDebugUtils`: Enables Validation Layers and forwards messages to spdlog.
- `ExtensionVkGLFW`: Initializes GLFW and creates window, surface.
- `ExtensionResources`: Convenience extension to initialize a `ResourceAllocator`, `MemoryAllocator` and `SamplerPool` using `MemoryAllocatorVMA`.
- `ExtensionVkFloatAtomics`: Enables the atomic float extension if available (`VK_EXT_shader_atomic_float`).
- `ExtensionVkRayTracingPipeline`: Enables ray tracing pipeline extensions and provides access to common functions.
- `ExtensionVkRayQuery`: Enables ray query extensions and provides access to common functions.
- `ExtensionVkAccelerationStructure`: Enables extensions to build ray tracing acceleration structures.
