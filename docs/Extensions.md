## Extensions

The Context can be extended by extensions. Extensions hook into the context creation process and enable Vulkan instance/device layers and extensions as well as features.

Extensions are registered by name in a global `ExtensionRegistry` and requested by name in `ContextCreateInfo::context_extensions`. The extensions marked auto-loaded below, the core `"merian"` extension among them, are loaded for every context.

Note: extensions can only be used after `Context` is initialized.

```c++
int main() {
    merian::ContextHandle context = merian::Context::create({
        // "merian", "merian-resources", GLFW and SDL3 are loaded automatically.
        // "merian-validation-layers" also pulls in "merian-debug-utils".
        .context_extensions = {"merian-validation-layers", "merian-host-windows"},
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

| String name | Class | Auto-loaded | Purpose |
|---|---|---|---|
| `"merian"` | `ExtensionMerian` | yes | Core requirements (sync2, push descriptors, ...). |
| `"merian-compatibility"` | `ExtensionCompatibility` | yes | Cross-vendor compatibility fixes. |
| `"merian-mitigations"` | `ExtensionMitigations` | yes | Vendor-specific GPU mitigations. |
| `"merian-resources"` | `ExtensionResources` | yes | `ResourceAllocator`, `MemoryAllocator`, `SamplerPool` and staging. |
| `"VulkanMemoryAllocator"` | `ExtensionVMA` | yes | Memory allocator provider using the Vulkan Memory Allocator. |
| `"merian-layer-settings"` | `ExtensionVkLayerSettings` | yes | Vulkan layer settings. |
| `"merian-debug-utils"` | `ExtensionVkDebugUtils` | debug builds | Debug utils messenger with spdlog forwarding. |
| `"merian-validation-layers"` | `ExtensionVkValidationLayers` | no | Khronos validation layer, with `"merian-debug-utils"`. |
| `"merian-device-fault"` | `ExtensionDeviceFault` | no | `VK_EXT_device_fault`, to make a device loss diagnosable. |
| `"merian-glslang-compiler"` | `ExtensionGlslangCompiler` | yes | GLSL runtime compilation with the glslang library (`-Dglslang`). |
| `"merian-glslangvalidator-compiler"` | `ExtensionGlslangValidatorCompiler` | yes | GLSL runtime compilation with `glslangValidator` from `PATH`. |
| `"merian-glslc-compiler"` | `ExtensionGlslcCompiler` | yes | GLSL runtime compilation with `glslc` from `PATH`. |
| `"merian-imgui"` | `ExtensionImGui` | yes | Features the ImGui renderer needs. |
| `"GLFW"` | `ExtensionGLFW` | yes (`-Dglfw`) | Window provider: GLFW, ensures present support on the GCT queue. |
| `"SDL3"` | `ExtensionSDL` | yes (`-Dsdl`) | Initializes SDL3. |
| `"SDL3 (Video Subsystem)"` | `ExtensionSDLVideo` | yes (`-Dsdl`) | Window provider: SDL3. |
| `"SDL3 (Audio Subsystem)"` | `ExtensionSDLAudio` | yes (`-Dsdl`) | Audio device provider: SDL3. |
| `"merian-host-windows"` | `ExtensionHostWindows` | no | Windows a host application shows, presenting into host memory (see [Window](Window.md)). |
| `"FFmpeg"` | `ExtensionFFmpeg` | no | Video device provider: hardware en-/decoding through FFmpeg (`-Dffmpeg`). |
| `"ngx"` | `ExtensionNGX` | no | NVIDIA NGX (`-Ddlss`). |
| `"dlss-super-resolution"` | `ExtensionDLSSSuperResolution` | no | DLSS super resolution, with `"ngx"`. |
| `"dlss-ray-reconstruction"` | `ExtensionDLSSRayReconstruction` | no | DLSS ray reconstruction, with `"ngx"`. |
| `"merian-graph"` | `MerianGraphExtension` | no | Node registry and graph creation (`merian-graph` library). |

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
