#include "merian/vk/extension/sdl/extension_sdl_window.hpp"
#include "merian/vk/extension/sdl/sdl_window.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <spdlog/spdlog.h>

namespace merian {

ExtensionSDLWindow::ExtensionSDLWindow() : ContextExtension() {
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) {
        SPDLOG_WARN("SDL_InitSubSystem(SDL_INIT_VIDEO) failed: {}", SDL_GetError());
        video_initialized = false;
        sdl_vulkan_support = false;
        return;
    }
    video_initialized = true;

    sdl_vulkan_support = SDL_Vulkan_LoadLibrary(nullptr);
    if (!sdl_vulkan_support)
        SPDLOG_WARN("SDL Vulkan support unavailable: {}", SDL_GetError());
    else
        SPDLOG_DEBUG("SDL video + Vulkan initialized");
}

ExtensionSDLWindow::~ExtensionSDLWindow() {
    if (sdl_vulkan_support)
        SDL_Vulkan_UnloadLibrary();
    if (video_initialized) {
        SPDLOG_DEBUG("Shutdown SDL video subsystem");
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
    }
}

std::vector<std::string> ExtensionSDLWindow::request_extensions() {
    return {"merian-sdl"};
}

InstanceSupportInfo
ExtensionSDLWindow::query_instance_support(const InstanceSupportQueryInfo& /*query_info*/) {
    InstanceSupportInfo info;
    info.supported = sdl_vulkan_support;

    if (sdl_vulkan_support) {
        Uint32 count = 0;
        const char* const* exts = SDL_Vulkan_GetInstanceExtensions(&count);
        if (exts) {
            info.required_extensions.assign(exts, exts + count);
        }
    }

    return info;
}

DeviceSupportInfo
ExtensionSDLWindow::query_device_support(const DeviceSupportQueryInfo& query_info) {
    if (!sdl_vulkan_support)
        return DeviceSupportInfo{false, "SDL Vulkan support unavailable"};

    // Check swapchain extension availability
    DeviceSupportInfo info =
        DeviceSupportInfo::check(query_info, {}, {}, {VK_KHR_SWAPCHAIN_EXTENSION_NAME});

    if (!info.supported)
        return info;

    // Verify presentation support for the selected queue family via SDL3's dedicated API
    const bool present_supported = SDL_Vulkan_GetPresentationSupport(
        **(query_info.physical_device->get_instance()), **query_info.physical_device,
        query_info.queue_info.queue_family_idx_GCT);
    if (!present_supported) {
        info.supported = false;
        info.unsupported_reason = "Queue family does not support SDL surface presentation";
    }

    return info;
}

bool ExtensionSDLWindow::accept_graphics_queue(const InstanceHandle& instance,
                                                const PhysicalDeviceHandle& physical_device,
                                                std::size_t queue_family_index) {
    return !sdl_vulkan_support ||
           SDL_Vulkan_GetPresentationSupport(**instance, **physical_device,
                                             static_cast<Uint32>(queue_family_index));
}

SDLWindowHandle ExtensionSDLWindow::create_window(const DeviceHandle& device,
                                                   const int width,
                                                   const int height,
                                                   const char* title) const {
    return std::shared_ptr<SDLWindow>(new SDLWindow(device, width, height, title));
}

} // namespace merian
