#include "merian/vk/extension/sdl/extension_sdl.hpp"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

namespace merian {

ExtensionSDL::ExtensionSDL() : ContextExtension() {
    SPDLOG_DEBUG("Initialize SDL {}...", SDL_GetRevision());
    if (!SDL_Init(0)) {
        SPDLOG_WARN("SDL_Init failed: {}", SDL_GetError());
        return;
    }
    sdl_initialized = true;
    SPDLOG_DEBUG("...success!");
}

ExtensionSDL::~ExtensionSDL() {
    if (sdl_initialized) {
        SPDLOG_DEBUG("Shutdown SDL");
        SDL_Quit();
    }
}

InstanceSupportInfo
ExtensionSDL::query_instance_support(const InstanceSupportQueryInfo& /*query_info*/) {
    return InstanceSupportInfo{sdl_initialized};
}

} // namespace merian
