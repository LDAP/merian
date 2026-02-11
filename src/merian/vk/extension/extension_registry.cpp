#include "merian/vk/extension/extension_registry.hpp"

#include "merian/vk/extension/extension_glfw.hpp"
#include "merian/vk/extension/extension_glsl_compiler.hpp"
#include "merian/vk/extension/extension_merian.hpp"
#include "merian/vk/extension/extension_mitigations.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_layer_settings.hpp"
#include "merian/vk/extension/extension_vma.hpp"

namespace merian {

ExtensionRegistry& ExtensionRegistry::get_instance() {
    static ExtensionRegistry instance;
    return instance;
}

ExtensionRegistry::ExtensionRegistry() {
    register_extension<ExtensionGLFW>("merian-glfw");
    register_extension<ExtensionGLSLCompiler>("merian-glsl-compiler");
    register_extension<ExtensionMerian>("merian");
    register_extension<ExtensionMitigations>("merian-mitigations");
    register_extension<ExtensionResources>("merian-resources");
    register_extension<ExtensionVkDebugUtils>("vk_debug_utils");
    register_extension<ExtensionVkLayerSettings>("vk_layer_settings");
    register_extension<ExtensionVMA>("merian-vma");
}

std::shared_ptr<ContextExtension> ExtensionRegistry::create(const std::string& name) const {
    auto it = name_to_factory.find(name);
    if (it != name_to_factory.end()) {
        return it->second();
    }
    return nullptr;
}

bool ExtensionRegistry::is_registered(const std::string& name) const {
    return name_to_factory.contains(name);
}

std::vector<std::string> ExtensionRegistry::get_registered_extensions() const {
    std::vector<std::string> names;
    names.reserve(name_to_factory.size());
    for (const auto& [name, _] : name_to_factory) {
        names.push_back(name);
    }
    return names;
}

} // namespace merian
