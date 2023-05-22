#pragma once

#include <GLFW/glfw3.h>
#include <vk/extension/extension.hpp>

class ExtensionGLFW : public Extension {
  public:
    ~ExtensionGLFW() {}
    std::string name() const {
        return "ExtensionGLFW";
    }
    std::vector<const char*> required_extension_names() const {
        std::vector<const char*> required_extensions;
        uint32_t count;
        const char** extensions = glfwGetRequiredInstanceExtensions(&count);
        required_extensions.insert(required_extensions.end(), extensions, extensions + count);
        return required_extensions;
    }
};
