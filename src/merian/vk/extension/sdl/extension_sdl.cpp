#include "merian/vk/extension/sdl/extension_sdl.hpp"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

namespace merian {

ExtensionSDL::ExtensionSDL() : ContextExtension() {}

ExtensionSDL::~ExtensionSDL() {
    if (sdl_initialized) {
        SPDLOG_DEBUG("Shutdown SDL");
        SDL_Quit();
    }
}

InstanceSupportInfo
ExtensionSDL::query_instance_support(const InstanceSupportQueryInfo& /*query_info*/) {
    if (!sdl_initialized) {
        SPDLOG_DEBUG("Initialize SDL {}...", SDL_GetRevision());
        if (!SDL_Init(0)) {
            SPDLOG_WARN("SDL_Init failed: {}", SDL_GetError());
        } else {
            sdl_initialized = true;
            SPDLOG_DEBUG("...success!");
        }
    }
    return InstanceSupportInfo{sdl_initialized};
}

} // namespace merian
