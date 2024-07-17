## Window creation

### GLFW

Merian provides an extension `ExtensionVkGLFW` that initializes GLFW. With the extension active a window can be simply constructed.


```c++

int main() {
    auto extGLFW = std::make_shared<merian::ExtensionVkGLFW>();
    merian::ContextHandle context = merian::Context::make_context({extGLFW}, "Quake");

    merian::GLFWWindowHandle window = std::make_shared<merian::GLFWWindow>(context);
    // Creates a corresponding Vulkan surface
    merian::SurfaceHandle surface = window->get_surface();

    // Do stuff with the window and surface.
    //...
    
    // everything is cleaned up automatically when the program exits.
}
```
