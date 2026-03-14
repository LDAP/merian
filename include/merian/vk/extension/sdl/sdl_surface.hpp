#pragma once

#include "merian/vk/extension/sdl/sdl_window.hpp"
#include "merian/vk/window/surface.hpp"

#include <SDL3/SDL_vulkan.h>
#include <spdlog/spdlog.h>

namespace merian {

inline vk::SurfaceKHR surface_from_sdl_window(const DeviceHandle& device,
                                               const SDLWindowHandle& window) {
    VkSurfaceKHR psurf;
    if (!SDL_Vulkan_CreateSurface(
            window->get_window(),
            device->get_physical_device()->get_instance()->get_instance(), nullptr, &psurf))
        throw std::runtime_error(fmt::format("SDL surface creation failed: {}", SDL_GetError()));
    return vk::SurfaceKHR(psurf);
}

class SDLSurface : public Surface {
  public:
    SDLSurface(const DeviceHandle& device, const SDLWindowHandle& window)
        : Surface(device, surface_from_sdl_window(device, window)), window(window) {
        SPDLOG_DEBUG("create surface ({})", fmt::ptr(this));
    }

  private:
    const SDLWindowHandle window;
};

} // namespace merian
