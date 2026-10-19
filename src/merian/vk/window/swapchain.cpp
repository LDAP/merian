#include "merian/vk/window/swapchain.hpp"

namespace {

vk::PresentModeKHR select_best(const std::vector<vk::PresentModeKHR>& available,
                               const std::vector<vk::PresentModeKHR>& preffered) {
    assert(!available.empty());

    for (const auto& p : preffered) {
        for (const auto& a : available) {
            if (p == a) {
                return a;
            }
        }
    }

    return available[0];
}

vk::SurfaceFormatKHR select_best(const std::vector<vk::SurfaceFormatKHR>& available,
                                 const std::vector<vk::SurfaceFormatKHR>& preffered) {
    assert(!available.empty());

    for (const auto& p : preffered) {
        for (const auto& a : available) {
            if (p.format == a.format) {
                return a;
            }
        }
    }

    return available[0];
}

} // namespace

namespace merian {

// -------------------------------------------------------------------------------------

SwapchainImage::SwapchainImage(const ContextHandle& context,
                               const vk::Image& image,
                               const vk::ImageCreateInfo create_info,
                               const SwapchainHandle& swapchain)
    : Image(context, image, create_info, swapchain->get_present_layout()), swapchain(swapchain) {}

SwapchainImage::~SwapchainImage() {
    // prevent image destruction, the swapchain destroys the image
    get_image() = VK_NULL_HANDLE;
}

// -------------------------------------------------------------------------------------

Swapchain::Swapchain(const ContextHandle& context,
                     const std::vector<vk::SurfaceFormatKHR>& supported_surface_formats,
                     const std::vector<vk::PresentModeKHR>& supported_present_modes,
                     const uint32_t min_images,
                     const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats,
                     const std::vector<vk::PresentModeKHR>& preferred_present_modes)
    : context(context), supported_present_modes(supported_present_modes),
      supported_surface_formats(supported_surface_formats), new_min_images(min_images) {
    assert(context);

    if (supported_surface_formats.empty())
        throw std::runtime_error("Surface doesn't support any surface formats!");
    if (supported_present_modes.empty())
        throw std::runtime_error("Surface doesn't support any present modes!");

    new_surface_format = select_best(supported_surface_formats, preferred_surface_formats);
    new_present_mode = select_best(supported_present_modes, preferred_present_modes);
}

Swapchain::Swapchain(const SwapchainHandle& previous)
    : context(previous->context), supported_present_modes(previous->supported_present_modes),
      supported_surface_formats(previous->supported_surface_formats),
      new_surface_format(previous->new_surface_format),
      new_present_mode(previous->new_present_mode), new_min_images(previous->new_min_images) {}

bool Swapchain::settings_changed() const {
    assert(info);
    return new_present_mode != info->present_mode || new_surface_format != info->surface_format;
}

// -------------------------------------------------------------------------------------

const std::vector<vk::PresentModeKHR>& Swapchain::get_supported_present_modes() const {
    return supported_present_modes;
}

const std::vector<vk::SurfaceFormatKHR>& Swapchain::get_supported_surface_formats() const {
    return supported_surface_formats;
}

// returns the actually selected one from supported_present_modes
vk::PresentModeKHR Swapchain::set_new_present_mode(const vk::PresentModeKHR desired) {
    new_present_mode = select_best(supported_present_modes, {desired, new_present_mode});

    return new_present_mode;
}

// returns the actually selected one from supported_surface_formats
vk::SurfaceFormatKHR Swapchain::set_new_surface_format(const vk::SurfaceFormatKHR desired) {
    new_surface_format = select_best(supported_surface_formats, {desired, new_surface_format});

    return new_surface_format;
}

void Swapchain::set_min_images(const uint32_t min_images) {
    new_min_images = min_images;
}

const vk::PresentModeKHR& Swapchain::get_new_present_mode() const {
    return new_present_mode;
}

const vk::SurfaceFormatKHR& Swapchain::get_new_surface_format() const {
    return new_surface_format;
}

} // namespace merian
