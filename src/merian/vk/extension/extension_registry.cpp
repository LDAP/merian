#include "merian/vk/extension/extension_registry.hpp"

#include "merian/shader/glsl_compiler_provider.hpp"
#include "merian/vk/extension/extension_compatibility.hpp"
#include "merian/vk/extension/extension_glslang_compiler.hpp"
#include "merian/vk/extension/extension_glslangvalidator_compiler.hpp"
#include "merian/vk/extension/extension_glslc_compiler.hpp"
#include "merian/vk/extension/extension_merian.hpp"
#include "merian/vk/extension/extension_mitigations.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_shaderc_compiler.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_layer_settings.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"
#include "merian/vk/extension/extension_vma.hpp"
#include "merian/vk/imgui/extension_imgui.hpp"
#include "merian/vk/memory/memory_allocator_provider.hpp"

#ifdef MERIAN_GLFW_ENABLED
#include "merian/vk/extension/glfw/extension_glfw.hpp"
#endif

#ifdef MERIAN_SDL_ENABLED
#include "merian/vk/extension/sdl/extension_sdl.hpp"
#include "merian/vk/extension/sdl/extension_sdl_audio.hpp"
#include "merian/vk/extension/sdl/extension_sdl_video.hpp"
#endif

namespace merian {

ExtensionRegistry& ExtensionRegistry::get_instance() {
    static ExtensionRegistry instance;
    return instance;
}

ExtensionRegistry::ExtensionRegistry() {
#ifdef MERIAN_GLFW_ENABLED
    register_extension<ExtensionGLFW>(ExtensionGLFW::name, true,
                                      {ProviderPriority<WindowProvider>{40}});
#endif
#ifdef MERIAN_SDL_ENABLED
    register_extension<ExtensionSDL>(ExtensionSDL::name, true);
    register_extension<ExtensionSDLAudio>(ExtensionSDLAudio::name, true,
                                          {ProviderPriority<AudioDeviceProvider>{50}});
    register_extension<ExtensionSDLVideo>(ExtensionSDLVideo::name, true,
                                          {ProviderPriority<WindowProvider>{50}});
#endif
    register_extension<ExtensionCompatibility>(ExtensionCompatibility::name, true);
    register_extension<ExtensionMitigations>(ExtensionMitigations::name, true);
    register_extension<ExtensionVkDebugUtils>(ExtensionVkDebugUtils::name, Context::IS_DEBUG_BUILD);
    register_extension<ExtensionGlslangCompiler>(ExtensionGlslangCompiler::name, true,
                                                 {ProviderPriority<GLSLCompilerProvider>{100}});
    register_extension<ExtensionGlslangValidatorCompiler>(
        ExtensionGlslangValidatorCompiler::name, true,
        {ProviderPriority<GLSLCompilerProvider>{80}});
    register_extension<ExtensionShadercCompiler>(ExtensionShadercCompiler::name, true,
                                                 {ProviderPriority<GLSLCompilerProvider>{60}});
    register_extension<ExtensionGlslcCompiler>(ExtensionGlslcCompiler::name, true,
                                               {ProviderPriority<GLSLCompilerProvider>{40}});
    register_extension<ExtensionMerian>(ExtensionMerian::name, true);
    register_extension<ExtensionVMA>(ExtensionVMA::name, true,
                                     {ProviderPriority<MemoryAllocatorProvider>{50}});
    register_extension<ExtensionResources>(ExtensionResources::name, true);
    register_extension<ExtensionVkLayerSettings>(ExtensionVkLayerSettings::name, true);
    register_extension<ExtensionVkValidationLayers>(ExtensionVkValidationLayers::name, false);
    register_extension<ExtensionImGui>(ExtensionImGui::name, true);
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

int ExtensionRegistry::get_priority(const std::string& name,
                                    const std::type_index& interface_type) const {
    auto ext_it = name_to_provider_priority.find(name);
    if (ext_it == name_to_provider_priority.end())
        return 0;
    auto iface_it = ext_it->second.find(interface_type);
    return iface_it != ext_it->second.end() ? iface_it->second : 0;
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
