#pragma once

#include "merian/utils/input_listener.hpp"
#include "merian/utils/vector_matrix.hpp"
#include "merian/vk/window/window.hpp"

#include <optional>

namespace merian {

// Picks pixels of an image presented through a Window's FIT blit: tracks the cursor, arms on
// modifier+LMB (consuming only that click so camera controls keep working), and maps window
// coordinates back to image pixels by inverting merian::fit. Input dispatch and polling both run
// on the graph thread. A presentation that is not FIT maps to the wrong pixel.
class ViewportPicker : public InputListener {
  public:
    // Register once per controller, between ImGui (10) and camera controls (0).
    static constexpr int DEFAULT_PRIORITY = 5;

    ViewportPicker(const InputController::ModKey modifier = InputController::ModKey::CONTROL)
        : modifier(modifier) {}

    bool on_cursor(InputController& controller, double xpos, double ypos) override;
    bool on_key(InputController& controller,
                InputController::Key key,
                InputController::KeyStatus action,
                int mods) override;
    bool on_mouse_button(InputController& controller,
                         InputController::MouseButton button,
                         InputController::KeyStatus status) override;
    bool on_scroll(InputController& controller, double xoffset, double yoffset) override;

    // Scroll steps accumulated while the modifier was held; clears the accumulator.
    double take_scroll();

    // The armed click mapped to image pixels, or nullopt; clears the click.
    std::optional<int2> take_click(const vk::Extent2D& image_extent, const WindowHandle& window);

    // Current cursor position mapped to image pixels; nullopt outside the image.
    std::optional<int2> cursor(const vk::Extent2D& image_extent, const WindowHandle& window) const;

  private:
    std::optional<int2> map_to_image(double x,
                                     double y,
                                     const vk::Extent2D& image_extent,
                                     const WindowHandle& window) const;

    const InputController::ModKey modifier;
    double cursor_x = 0.;
    double cursor_y = 0.;
    double click_x = 0.;
    double click_y = 0.;
    bool modifier_down = false;
    double scroll_accum = 0.;
    bool clicked = false;
    bool swallow_release = false;
};

} // namespace merian
