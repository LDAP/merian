#include "merian/vk/extension/sdl/extension_sdl_audio.hpp"
#include "merian/vk/extension/sdl/extension_sdl.hpp"
#include "merian/vk/extension/sdl/sdl_audio_device.hpp"

#include <SDL3/SDL.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace merian {

ExtensionSDLAudio::ExtensionSDLAudio() : ContextExtension() {}

ExtensionSDLAudio::~ExtensionSDLAudio() {
    if (audio_initialized) {
        SPDLOG_DEBUG("Shutdown SDL audio subsystem");
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
    }
}

std::vector<std::string> ExtensionSDLAudio::request_extensions() {
    return {ExtensionSDL::name};
}

InstanceSupportInfo
ExtensionSDLAudio::query_instance_support(const InstanceSupportQueryInfo& query_info) {
    sdl_ext = query_info.extension_container.get_context_extension<ExtensionSDL>(true);
    if (!sdl_ext)
        return InstanceSupportInfo{false, fmt::format("{} not available", ExtensionSDL::name)};

    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        SPDLOG_WARN("SDL_InitSubSystem(SDL_INIT_AUDIO) failed: {}", SDL_GetError());
        audio_initialized = false;
    } else {
        audio_initialized = true;
        SPDLOG_DEBUG("SDL audio subsystem initialized");
    }
    return InstanceSupportInfo{audio_initialized};
}

AudioDeviceHandle ExtensionSDLAudio::create_audio_device() const {
    return AudioDeviceHandle(new SDLAudioDevice());
}

} // namespace merian
