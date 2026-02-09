#include "merian/vk/extension/extension_registry.hpp"

#include "merian/vk/extension/extension_glfw.hpp"
#include "merian/vk/extension/extension_merian.hpp"
#include "merian/vk/extension/extension_mitigations.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_layer_settings.hpp"

namespace merian {

ExtensionRegistry& ExtensionRegistry::get_instance() {
    static ExtensionRegistry instance;
    return instance;
}

ExtensionRegistry::ExtensionRegistry() {
    register_extension("glfw", create_extension<ExtensionGLFW>);
    register_extension("merian", create_extension<ExtensionMerian>);
    register_extension("mitigations", create_extension<ExtensionMitigations>);
    register_extension("resources", create_extension<ExtensionResources>);
    register_extension("vk_debug_utils", create_extension<ExtensionVkDebugUtils>);
    register_extension("vk_layer_settings", create_extension<ExtensionVkLayerSettings>);
}

void ExtensionRegistry::register_extension(const std::string& name,
                                           const ExtensionFactory& factory) {
    registry[name] = factory;
}

std::shared_ptr<ContextExtension> ExtensionRegistry::create(const std::string& name) const {
    auto it = registry.find(name);
    if (it != registry.end()) {
        return it->second();
    }
    return nullptr;
}

bool ExtensionRegistry::is_registered(const std::string& name) const {
    return registry.contains(name);
}

std::vector<std::string> ExtensionRegistry::get_registered_extensions() const {
    std::vector<std::string> names;
    names.reserve(registry.size());
    for (const auto& [name, _] : registry) {
        names.push_back(name);
    }
    return names;
}

} // namespace merian
