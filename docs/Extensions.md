## Extensions

The Context can be extended by extensions. Extensions hook into the context creation process and enable Vulkan instance/device layers and extensions as well as features.

Extensions are registered by name in a global `ExtensionRegistry` and requested by name in `ContextCreateInfo::context_extensions`. The core `"merian"` extension is always loaded automatically and pulls in mitigations and compatibility fixes (and debug utils in debug builds).

Note: extensions can only be used after `Context` is initialized.

```c++
int main() {
    merian::ContextHandle context = merian::Context::create({
        // "merian" is loaded automatically — no need to list it.
        // "merian-resources" provides ResourceAllocator, MemoryAllocator and SamplerPool.
        .context_extensions = {"merian-resources", "merian-glfw", "merian-debug-utils"},
    });

    // After creation, retrieve extensions via:
    auto resources = context->get_context_extension<merian::ExtensionResources>();
    // Returns nullptr if the extension is not active or unsupported.
}
```

### Naming conventions

- `Extension*`: General name for an extension class.
- `ExtensionVk*`: Extension that enables/requires Vulkan extensions or features.

### Built-in extensions

| String name | Class | Purpose |
|---|---|---|
| `"merian"` | `ExtensionMerian` | **Always auto-loaded.** Core requirements (sync2, push descriptors, mitigations, compatibility). |
| `"merian-resources"` | `ExtensionResources` | `ResourceAllocator`, `MemoryAllocator` and `SamplerPool` via VMA. |
| `"merian-glfw"` | `ExtensionGLFW` | Initializes GLFW, ensures present support on the GCT queue. |
| `"merian-debug-utils"` | `ExtensionVkDebugUtils` | Validation layers with spdlog forwarding. Auto-loaded by `"merian"` in debug builds. |
| `"merian-glsl-compiler"` | `ExtensionGLSLCompiler` | GLSL runtime shader compilation (glslang/glslc backend). |
| `"merian-validation-layers"` | `ExtensionVkValidationLayers` | Khronos validation layer. |
| `"merian-layer-settings"` | `ExtensionVkLayerSettings` | Vulkan layer settings. |
| `"merian-vma"` | `ExtensionVMA` | Vulkan Memory Allocator. Auto-pulled by `"merian-resources"`. |
| `"merian-mitigations"` | `ExtensionMitigations` | Vendor-specific GPU mitigations. Auto-pulled by `"merian"`. |
| `"merian-compatibility"` | `ExtensionCompatibility` | Cross-vendor compatibility fixes. Auto-pulled by `"merian"`. |

### Enabling Vulkan features

Vulkan features (e.g. ray tracing, float atomics) are enabled via `ContextCreateInfo::features` using a `VulkanFeatures` object, **not** via merian extensions:

```c++
merian::VulkanFeatures features;
features.set_feature("accelerationStructure", true);
features.set_feature("rayQuery", true);
features.set_feature("rayTracingPipeline", true);
features.set_feature("shaderAtomicFloat", true);

// or construct from a list of feature names:
merian::VulkanFeatures features{{"rayQuery", "accelerationStructure"}};

merian::ContextHandle context = merian::Context::create({
    .features = features,
    .context_extensions = {"merian-resources"},
});
```

### Enabling raw Vulkan extensions

Raw Vulkan device or instance extensions are passed via `ContextCreateInfo::device_extensions` / `ContextCreateInfo::instance_extensions`:

```c++
merian::ContextHandle context = merian::Context::create({
    .device_extensions = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME},
    .context_extensions = {"merian-resources"},
});
```

### Registering custom extensions

```c++
merian::ExtensionRegistry::get_instance()
    .register_extension<MyExtension>("my-extension-name");

// Then use it like any other:
merian::Context::create({ .context_extensions = {"my-extension-name"} });
```
