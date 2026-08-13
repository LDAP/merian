#include "merian/vk/window/viewport_picker.hpp"

#include "merian/vk/utils/math.hpp"

namespace merian {

bool ViewportPicker::on_cursor([[maybe_unused]] InputController& controller,
                               const double xpos,
                               const double ypos) {
    cursor_x = xpos;
    cursor_y = ypos;
    return false;
}

bool ViewportPicker::on_key([[maybe_unused]] InputController& controller,
                            const InputController::Key key,
                            const InputController::KeyStatus action,
                            [[maybe_unused]] const int mods) {
    const bool is_modifier =
        (modifier == InputController::ModKey::CONTROL &&
         (key == InputController::Key::LEFT_CTRL || key == InputController::Key::RIGHT_CTRL)) ||
        (modifier == InputController::ModKey::SHIFT &&
         (key == InputController::Key::LEFT_SHIFT || key == InputController::Key::RIGHT_SHIFT)) ||
        (modifier == InputController::ModKey::ALT &&
         (key == InputController::Key::LEFT_ALT || key == InputController::Key::RIGHT_ALT));
    if (is_modifier) {
        modifier_down = action == InputController::KeyStatus::PRESS ||
                        action == InputController::KeyStatus::REPEAT;
    }
    return false;
}

bool ViewportPicker::on_mouse_button([[maybe_unused]] InputController& controller,
                                     const InputController::MouseButton button,
                                     const InputController::KeyStatus status) {
    if (button != InputController::MouseButton::MOUSE1) {
        return false;
    }
    if (status == InputController::KeyStatus::PRESS && modifier_down) {
        click_x = cursor_x;
        click_y = cursor_y;
        clicked = true;
        swallow_release = true;
        return true;
    }
    if (status == InputController::KeyStatus::RELEASE && swallow_release) {
        swallow_release = false;
        return true;
    }
    return false;
}

std::optional<int2> ViewportPicker::map_to_image(const double x,
                                                 const double y,
                                                 const vk::Extent2D& image_extent,
                                                 const WindowHandle& window) const {
    if (!window || image_extent.width == 0 || image_extent.height == 0) {
        return std::nullopt;
    }
    const vk::Extent2D fb = window->framebuffer_extent();
    if (fb.width == 0 || fb.height == 0) {
        return std::nullopt;
    }

    // invert the very transform the presenting blit used, so the rounding matches
    const auto [lower, upper] =
        fit(vk::Offset3D{0, 0, 0},
            vk::Offset3D{static_cast<int32_t>(image_extent.width),
                         static_cast<int32_t>(image_extent.height), 1},
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{static_cast<int32_t>(fb.width), static_cast<int32_t>(fb.height), 1});
    const float width = static_cast<float>(upper.x - lower.x);
    const float height = static_cast<float>(upper.y - lower.y);
    if (width <= 0.f || height <= 0.f) {
        return std::nullopt;
    }

    const float density = window->get_pixel_density();
    const float u = ((static_cast<float>(x) * density) - static_cast<float>(lower.x)) / width;
    const float v = ((static_cast<float>(y) * density) - static_cast<float>(lower.y)) / height;
    if (u < 0.f || v < 0.f || u >= 1.f || v >= 1.f) {
        return std::nullopt;
    }
    return int2(static_cast<int32_t>(u * static_cast<float>(image_extent.width)),
                static_cast<int32_t>(v * static_cast<float>(image_extent.height)));
}

std::optional<int2> ViewportPicker::take_click(const vk::Extent2D& image_extent,
                                               const WindowHandle& window) {
    if (!clicked) {
        return std::nullopt;
    }
    clicked = false;
    return map_to_image(click_x, click_y, image_extent, window);
}

std::optional<int2> ViewportPicker::cursor(const vk::Extent2D& image_extent,
                                           const WindowHandle& window) const {
    return map_to_image(cursor_x, cursor_y, image_extent, window);
}

} // namespace merian
