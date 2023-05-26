# Merian Vulkan Framework


## Usage

### Context
The `Context` class initializes and destroys a Vulkan device and holds core objects (PhysicalDevice, Device, Queues, ...).

### Extensions
The Context can be extended by Extensions. The extensions can hook into the context creation process and enable Vulkan instance/device layers and extensions as well as features.

#### Naming conventions:

- Extension*: General name of an extension
- ExtensionVk*: Name of an extension that enables / requires / is associated with Vulkan extensions or features.

#### Current extensions:

Common:
- `ExtensionDebugUtils`: Enables Validation Layers.
- `ExtensionGLFW`: Initializes GLFW and creates window, surface and swapchain.

Features:
- `ExtensionFloatAtomics`: Enables float atomics (`VK_EXT_shader_atomic_float`).
- `ExtensionV12`: Enables Vulkan 1.2 features.

Ray tracing:
- `ExtensionVkRayTracingPipeline`: Enables ray tracing pipeline extensions and provides access to common functions.
- `ExtensionVkRayQuery`: Enables ray query extensions and provides access to common functions.
- `ExtensionVkAccelerationStructure`: Build and manage ray tracing acceleration structures extensions and provides access to common functions.


## Building

```bash
meson setup build
# or to compile with debug flags
meson setup build --buildtype=debug

meson compile -C build
```

