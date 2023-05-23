#include "vk/extension/extension.hpp"
class ExtensionFloatAtomics : public Extension {
  public:
    ExtensionFloatAtomics() {}
    ~ExtensionFloatAtomics() {}
    std::string name() const override {
        return "ExtensionFloatAtomics";
    }
    std::vector<const char*> required_device_extension_names() const override {
        return {
            VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        };
    }
};
