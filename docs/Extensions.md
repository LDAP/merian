## Extensions

The Context can be extended by Extensions.
The extensions can hook into the context creation process and enable Vulkan instance/device layers and extensions as well as features.

Note: Extension can only be used after `Context` is initialized.

```c++

int main() {
    const auto debug_utils = std::make_shared<merian::ExtensionVkDebugUtils>(false);
    const auto extGLFW = std::make_shared<merian::ExtensionVkGLFW>();
    const auto resources = std::make_shared<merian::ExtensionResources>();
    const std::vector<std::shared_ptr<merian::Extension>> extensions = {extGLFW, resources, debug_utils};

    const merian::SharedContext context = merian::Context::make_context(extensions, "merian");

    // use extensions...
}
```

### Naming conventions:

- `Extension*`: General name of an extension
- `ExtensionVk*`: Name of an extension that enables / requires / is associated with Vulkan extensions or features.

### Current extensions:

Common:

- `ExtensionResources`: Convenience extension to initialize a `ResourceAllocator`, `MemoryAllocator` and `SamplerPool` using `MemoryAllocatorVMA`.
- `ExtensionVkAccelerationStructure`: Enables extensions to build ray tracing acceleration structures.
- `ExtensionVkDebugUtils`: Enables Validation Layers and forwards messages to spdlog.
- `ExtensionVkFloatAtomics`: Enables the atomic float extension if available (`VK_EXT_shader_atomic_float`).
- `ExtensionVkGLFW`: Initializes GLFW and ensures present support on the graphics queue.
- `ExtensionVkRayQuery`: Enables ray query extensions and provides access to common functions.
- `ExtensionVkRayTracingPipeline`: Enables ray tracing pipeline extensions and provides access to common functions.
