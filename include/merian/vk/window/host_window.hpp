#pragma once

#include "merian/utils/input_controller.hpp"
#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/utils/host_copy.hpp"
#include "merian/vk/window/window.hpp"

#include <deque>
#include <functional>
#include <mutex>
#include <optional>

namespace merian {

// A window a host application shows. The host functions are thread-safe; the input, the extent
// and the display scale apply in poll_events().
class HostWindow : public Window, public InputController {
  public:
    explicit HostWindow(const WindowCreateInfo& create_info = {});

    vk::Extent2D framebuffer_extent() override;
    vk::Extent2D window_extent() override;
    SwapchainHandle create_swapchain(const ContextHandle& context) override;
    bool should_close() const override;
    void poll_events() override;
    bool is_fullscreen() const override;
    void set_fullscreen(const bool fullscreen) override;
    void set_size(const uint32_t width, const uint32_t height) override;
    void set_title(const std::string& title) override;
    InputControllerHandle get_input_controller() override;
    float get_display_scale() override;
    void set_cursor(WindowCursorShape shape) override;
    bool set_mouse_grabbed(bool grabbed) override;
    bool is_mouse_grabbed() const override;
    void set_clipboard_text(const char* text) override;
    const char* get_clipboard_text() override;

    // In pixels, at a display scale (pixels per logical pixel, > 0).
    void set_extent(const vk::Extent2D framebuffer_extent, const float display_scale = 1.f);

    void request_close();

    std::string get_title() const;

    WindowCursorShape get_cursor() const;

    // The newest frame that arrived since the last call.
    std::optional<HostCopy> get_new_frame();

    // In pixels; listeners get window coordinates, the pixels over the display scale.
    void send_cursor(const double x, const double y);
    void send_mouse_button(const MouseButton button, const KeyStatus status, const int mods);
    void send_scroll(const double x, const double y);
    void send_key(const Key key, const KeyStatus status, const int mods);
    void send_char(const unsigned int codepoint);
    void send_focus(const bool focused);

  private:
    friend class HostSwapchain;

    struct Presented {
        HostCopy frame;
        TimelineSemaphoreHandle semaphore;
        uint64_t semaphore_value;

        bool arrived() const;
    };

    void send(std::function<void()>&& dispatch, const bool cursor = false);

    void add_presented(const HostCopy& copy,
                       const TimelineSemaphoreHandle& semaphore,
                       const uint64_t semaphore_value);

    mutable std::mutex mutex;

    std::string title;
    vk::Extent2D extent;
    float display_scale = 1.f;
    vk::Extent2D dispatched_extent;
    float dispatched_display_scale = 1.f;
    bool close_requested = false;
    bool mouse_grabbed = false;
    WindowCursorShape cursor = WindowCursorShape::Arrow;
    std::string clipboard;

    std::vector<std::function<void()>> input;
    bool input_ends_with_cursor = false;
    // The newest frame that arrived, if any, then those in flight
    std::deque<Presented> presented;
};

using HostWindowHandle = std::shared_ptr<HostWindow>;

} // namespace merian
