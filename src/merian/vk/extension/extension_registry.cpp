#include "merian/vk/extension/extension_registry.hpp"

#include "merian/vk/extension/extension_compatibility.hpp"
#include "merian/vk/extension/glfw/extension_glfw.hpp"
#include "merian/vk/extension/sdl/extension_sdl.hpp"
#include "merian/vk/extension/sdl/extension_sdl_audio.hpp"
#include "merian/vk/extension/sdl/extension_sdl_window.hpp"
#include "merian/vk/extension/extension_glsl_compiler.hpp"
#include "merian/vk/extension/extension_merian.hpp"
#include "merian/vk/extension/extension_mitigations.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_layer_settings.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"
#include "merian/vk/extension/extension_vma.hpp"

namespace merian {

ExtensionRegistry& ExtensionRegistry::get_instance() {
    static ExtensionRegistry instance;
    return instance;
}

ExtensionRegistry::ExtensionRegistry() {
    register_extension<ExtensionGLFW>(ExtensionGLFW::name);
    register_extension<ExtensionSDL>(ExtensionSDL::name);
    register_extension<ExtensionSDLAudio>(ExtensionSDLAudio::name);
    register_extension<ExtensionSDLWindow>(ExtensionSDLWindow::name);
    register_extension<ExtensionCompatibility>(ExtensionCompatibility::name);
    register_extension<ExtensionGLSLCompiler>(ExtensionGLSLCompiler::name);
    register_extension<ExtensionMerian>(ExtensionMerian::name);
    register_extension<ExtensionMitigations>(ExtensionMitigations::name);
    register_extension<ExtensionResources>(ExtensionResources::name);
    register_extension<ExtensionVkDebugUtils>(ExtensionVkDebugUtils::name);
    register_extension<ExtensionVkLayerSettings>(ExtensionVkLayerSettings::name);
    register_extension<ExtensionVkValidationLayers>(ExtensionVkValidationLayers::name);
    register_extension<ExtensionVMA>(ExtensionVMA::name);
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
