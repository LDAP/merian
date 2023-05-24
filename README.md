# Merian Vulkan Framework


## Usage

### Context
The `Context` class initializes and destroys a Vulkan device and holds core objects (PhysicalDevice, Device, Queues, ...).

### Extensions
The Context can be extended by Extensions. The extensions can hook into the context creation process and enable Vulkan instance/device layers and extensions as well as features.

- `ExtensionDebugUtils`: Enables Validation Layers.
- `ExtensionFloatAtomics`: Enables float atomics (`VK_EXT_shader_atomic_float`).
- `ExtensionGLFW`: Initializes GLFW and creates window, surface and swapchain.
- `ExtensionRaytrace`: Enables ray tracing (ray query) extensions and provides access to common functions.
- `ExtensionV12`: Enables Vulkan 1.2 features.


## Building

```bash
meson setup build
# or to compile with debug flags
meson setup build --buildtype=debug

meson compile -C build
```
