#include "merian/vk/extension/sdl/extension_sdl_video.hpp"
#include "merian/vk/extension/sdl/extension_sdl.hpp"
#include "merian/vk/extension/sdl/sdl_window.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace merian {

ExtensionSDLVideo::ExtensionSDLVideo() : ContextExtension() {}

ExtensionSDLVideo::~ExtensionSDLVideo() {
    if (sdl_vulkan_support)
        SDL_Vulkan_UnloadLibrary();
    if (video_initialized) {
        SPDLOG_DEBUG("Shutdown SDL video subsystem");
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
    }
}

std::vector<std::string> ExtensionSDLVideo::request_extensions() {
    return {ExtensionSDL::name};
}

InstanceSupportInfo
ExtensionSDLVideo::query_instance_support(const InstanceSupportQueryInfo& query_info) {
    sdl_ext = query_info.extension_container.get_context_extension<ExtensionSDL>(true);
    if (!sdl_ext)
        return InstanceSupportInfo{false, fmt::format("{} not available", ExtensionSDL::name)};

    if (!video_initialized) {
        if (getenv("WAYLAND_DISPLAY") != nullptr) {
            SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
        }
        if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) {
            SPDLOG_WARN("SDL_InitSubSystem(SDL_INIT_VIDEO) failed: {}", SDL_GetError());
            return InstanceSupportInfo{false, "SDL_InitSubSystem(SDL_INIT_VIDEO) failed"};
        }
        video_initialized = true;

        sdl_vulkan_support = SDL_Vulkan_LoadLibrary(nullptr);
        if (!sdl_vulkan_support)
            SPDLOG_WARN("SDL Vulkan support unavailable: {}", SDL_GetError());
        else
            SPDLOG_DEBUG("SDL video + Vulkan initialized");
    }

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
ExtensionSDLVideo::query_device_support(const DeviceSupportQueryInfo& query_info) {
    if (!sdl_vulkan_support)
        return DeviceSupportInfo{false, "SDL Vulkan support unavailable"};

    DeviceSupportInfo info =
        DeviceSupportInfo::check(query_info, {}, {}, {VK_KHR_SWAPCHAIN_EXTENSION_NAME});

    if (!info.supported)
        return info;

    const bool present_supported = SDL_Vulkan_GetPresentationSupport(
        **(query_info.physical_device->get_instance()), **query_info.physical_device,
        query_info.queue_info.queue_family_idx_GCT);
    if (!present_supported) {
        info.supported = false;
        info.unsupported_reason = "Queue family does not support SDL surface presentation";
    }

    return info;
}

bool ExtensionSDLVideo::accept_graphics_queue(const InstanceHandle& instance,
                                              const PhysicalDeviceHandle& physical_device,
                                              std::size_t queue_family_index) {
    return !sdl_vulkan_support ||
           SDL_Vulkan_GetPresentationSupport(**instance, **physical_device,
                                             static_cast<Uint32>(queue_family_index));
}

WindowHandle ExtensionSDLVideo::create_window(const DeviceHandle& device,
                                              const WindowCreateInfo& create_info) const {
    return std::shared_ptr<SDLWindow>(
        new SDLWindow(device, create_info.width, create_info.height, create_info.title.c_str()));
}

} // namespace merian
