#include "merian/vk/window/host_window.hpp"

#include "merian/vk/window/host_swapchain.hpp"

#include <cmath>
#include <utility>

namespace merian {

namespace {

vk::Extent2D logical_extent(const vk::Extent2D extent, const float display_scale) {
    return {static_cast<uint32_t>(std::lround(static_cast<float>(extent.width) / display_scale)),
            static_cast<uint32_t>(std::lround(static_cast<float>(extent.height) / display_scale))};
}

} // namespace

HostWindow::HostWindow(const WindowCreateInfo& create_info) : title(create_info.title) {}

vk::Extent2D HostWindow::framebuffer_extent() {
    const std::lock_guard lock(mutex);
    return dispatched_extent;
}

vk::Extent2D HostWindow::window_extent() {
    const std::lock_guard lock(mutex);
    return logical_extent(dispatched_extent, dispatched_display_scale);
}

SwapchainHandle HostWindow::create_swapchain(const ContextHandle& context) {
    return std::make_shared<HostSwapchain>(
        context, std::static_pointer_cast<HostWindow>(shared_from_this()));
}

bool HostWindow::should_close() const {
    const std::lock_guard lock(mutex);
    return close_requested;
}

void HostWindow::poll_events() {
    std::unique_lock lock(mutex);
    const std::vector<std::function<void()>> dispatches = std::exchange(input, {});
    input_ends_with_cursor = false;
    const vk::Extent2D current_extent = extent;
    const float current_display_scale = display_scale;
    const bool rescaled = std::exchange(dispatched_display_scale, display_scale) != display_scale;
    const bool resized = std::exchange(dispatched_extent, extent) != extent;
    lock.unlock();

    if (rescaled) {
        dispatch_display_scale_changed(current_display_scale);
    }
    if (resized) {
        dispatch_resize(current_extent, logical_extent(current_extent, current_display_scale));
    }
    for (const auto& dispatch : dispatches) {
        dispatch();
    }
}

bool HostWindow::is_fullscreen() const {
    return false;
}

void HostWindow::set_fullscreen(const bool /*fullscreen*/) {}

void HostWindow::set_size(const uint32_t /*width*/, const uint32_t /*height*/) {}

void HostWindow::set_title(const std::string& title) {
    const std::lock_guard lock(mutex);
    this->title = title;
}

InputControllerHandle HostWindow::get_input_controller() {
    return std::static_pointer_cast<InputController>(
        std::static_pointer_cast<HostWindow>(shared_from_this()));
}

float HostWindow::get_display_scale() {
    const std::lock_guard lock(mutex);
    return dispatched_display_scale;
}

void HostWindow::set_cursor(const WindowCursorShape shape) {
    const std::lock_guard lock(mutex);
    cursor = shape;
}

bool HostWindow::set_mouse_grabbed(const bool grabbed) {
    const std::lock_guard lock(mutex);
    mouse_grabbed = grabbed;
    return true;
}

bool HostWindow::is_mouse_grabbed() const {
    const std::lock_guard lock(mutex);
    return mouse_grabbed;
}

void HostWindow::set_clipboard_text(const char* text) {
    const std::lock_guard lock(mutex);
    clipboard = text;
}

const char* HostWindow::get_clipboard_text() {
    const std::lock_guard lock(mutex);
    return clipboard.empty() ? nullptr : clipboard.c_str();
}

void HostWindow::set_extent(const vk::Extent2D framebuffer_extent, const float display_scale) {
    if (!(display_scale > 0.f)) {
        throw std::invalid_argument{"the display scale must be positive"};
    }
    const std::lock_guard lock(mutex);
    extent = framebuffer_extent;
    this->display_scale = display_scale;
}

void HostWindow::request_close() {
    const std::lock_guard lock(mutex);
    close_requested = true;
}

std::string HostWindow::get_title() const {
    const std::lock_guard lock(mutex);
    return title;
}

WindowCursorShape HostWindow::get_cursor() const {
    const std::lock_guard lock(mutex);
    return cursor;
}

std::optional<HostCopy> HostWindow::get_new_frame() {
    const std::lock_guard lock(mutex);
    std::optional<HostCopy> frame;
    while (!presented.empty() && presented.front().arrived()) {
        frame = std::move(presented.front().frame);
        presented.pop_front();
    }
    return frame;
}

void HostWindow::send_cursor(const double x, const double y) {
    send([this, x,
          y] { dispatch_cursor(x / dispatched_display_scale, y / dispatched_display_scale); },
         true);
}

void HostWindow::send_mouse_button(const MouseButton button,
                                   const KeyStatus status,
                                   const int mods) {
    send([this, button, status, mods] {
        dispatch_key(Key::UNKNOWN, KeyStatus::UNKNOWN, mods);
        dispatch_mouse_button(button, status);
    });
}

void HostWindow::send_scroll(const double x, const double y) {
    send([this, x, y] { dispatch_scroll(x, y); });
}

void HostWindow::send_key(const Key key, const KeyStatus status, const int mods) {
    send([this, key, status, mods] { dispatch_key(key, status, mods); });
}

void HostWindow::send_char(const unsigned int codepoint) {
    send([this, codepoint] { dispatch_char(codepoint); });
}

void HostWindow::send_focus(const bool focused) {
    send([this, focused] { dispatch_focus_changed(focused); });
}

void HostWindow::send(std::function<void()>&& dispatch, const bool cursor) {
    const std::lock_guard lock(mutex);
    if (cursor && input_ends_with_cursor) {
        input.back() = std::move(dispatch);
    } else {
        input.emplace_back(std::move(dispatch));
    }
    input_ends_with_cursor = cursor;
}

void HostWindow::add_presented(const HostCopy& copy,
                               const TimelineSemaphoreHandle& semaphore,
                               const uint64_t semaphore_value) {
    const std::lock_guard lock(mutex);
    while (presented.size() > 1 && presented[1].arrived()) {
        presented.pop_front();
    }
    presented.emplace_back(copy, semaphore, semaphore_value);
}

bool HostWindow::Presented::arrived() const {
    return semaphore->get_counter_value() >= semaphore_value;
}

} // namespace merian
