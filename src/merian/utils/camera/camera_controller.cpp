#include "merian/utils/camera/camera_controller.hpp"

#include "merian/utils/properties.hpp"

namespace merian {

using Key = InputController::Key;
using KeyStatus = InputController::KeyStatus;
using MouseButton = InputController::MouseButton;

bool CameraController::on_key(InputController& /*controller*/,
                              const Key key,
                              const KeyStatus action,
                              const int mods) {
    if (action == KeyStatus::UNKNOWN) {
        return false;
    }
    const bool pressed = action != KeyStatus::RELEASE;
    key_fast = (mods & InputController::SHIFT) != 0;
    key_pan = (mods & InputController::SHIFT) != 0;

    switch (key) {
    case Key::W:
        key_forward = pressed;
        break;
    case Key::S:
        key_back = pressed;
        break;
    case Key::A:
        key_left = pressed;
        break;
    case Key::D:
        key_right = pressed;
        break;
    case Key::SPACE:
        key_up = pressed;
        break;
    case Key::C:
        key_down = pressed;
        break;
    default:
        return false;
    }
    // WASD is a fly-only binding; only claim the key while flying so it stays free otherwise.
    return right_held;
}

bool CameraController::on_mouse_button(InputController& controller,
                                       const MouseButton button,
                                       const KeyStatus status) {
    const bool pressed = status == KeyStatus::PRESS;
    if (button == MouseButton::MOUSE2) {
        right_held = pressed; // fly
    } else if (button == MouseButton::MOUSE1) {
        left_held = pressed; // orbit / pan
    } else if (button == MouseButton::MOUSE3) {
        middle_held = pressed; // orbit / pan
    } else {
        return false;
    }
    controller.set_mouse_grabbed(right_held || left_held || middle_held);
    have_cursor = false;
    return true;
}

bool CameraController::on_cursor(InputController& /*controller*/,
                                 const double xpos,
                                 const double ypos) {
    if (!right_held && !left_held && !middle_held) {
        return false;
    }
    if (have_cursor) {
        const float dx = static_cast<float>(xpos - last_x);
        const float dy = static_cast<float>(ypos - last_y);
        if (right_held) {
            target_camera.rotate(dx * look_sensitivity, -dy * look_sensitivity);
        } else if (key_pan) {
            // Scale by the target distance so panning keeps pace at any zoom.
            const float distance =
                merian::length(target_camera.get_position() - target_camera.get_target());
            target_camera.move(-dx * pan_sensitivity * distance, dy * pan_sensitivity * distance,
                               0.0f);
        } else {
            // Orbit moves the eye around the target, so the vertical sign is the opposite of look.
            target_camera.orbit(dx * orbit_sensitivity, dy * orbit_sensitivity);
        }
    }
    last_x = xpos;
    last_y = ypos;
    have_cursor = true;
    return true;
}

bool CameraController::on_scroll(InputController& /*controller*/,
                                 const double /*xoffset*/,
                                 const double yoffset) {
    if (right_held) {
        move_speed *= yoffset > 0.0 ? 1.1f : 0.9f;
    } else {
        // A fraction of the current distance keeps the zoom speed proportional at any range.
        const float distance =
            merian::length(target_camera.get_position() - target_camera.get_target());
        target_camera.dolly(-static_cast<float>(yoffset) * dolly_fraction * distance);
    }
    return true;
}

void CameraController::attach(const CameraHandle& camera) {
    if (camera == attached) {
        return;
    }
    attached = camera;
    target_camera = *camera;
    animator.snap(*camera);
}

void CameraController::update(const double dt_seconds) {
    const float right = static_cast<float>(key_right) - static_cast<float>(key_left);
    const float up = static_cast<float>(key_up) - static_cast<float>(key_down);
    const float back = static_cast<float>(key_back) - static_cast<float>(key_forward);
    if (right_held && (right != 0.0f || up != 0.0f || back != 0.0f)) {
        const float3 step = normalize(float3{right, up, back}) *
                            static_cast<float>(move_speed * (key_fast ? 4.0 : 1.0) * dt_seconds);
        target_camera.fly(step.x, step.y, step.z);
    }

    if (!attached) {
        return;
    }
    // Orbit eases toward the input target; fly tracks it exactly.
    if (is_smoothed()) {
        animator.follow(target_camera, dt_seconds);
    } else {
        animator.snap(target_camera);
    }
    const Camera& c = animator.get_current_camera();
    attached->look_at(c.get_position(), c.get_target(), c.get_up(), c.get_field_of_view_vertical());
}

void CameraController::properties(Properties& config) {
    config.config_float("look sensitivity", look_sensitivity, "fly-mode mouse look (rad/pixel)",
                        0.0001f, 0.0f);
    config.config_float("orbit sensitivity", orbit_sensitivity, "orbit drag (rad/pixel)", 0.0001f,
                        0.0f);
    config.config_float("pan sensitivity", pan_sensitivity, "pan drag (fraction of distance/pixel)",
                        0.0001f, 0.0f);
    config.config_float("move speed", move_speed, "fly WASD speed (units/s)", 0.05f, 0.0f);
}

} // namespace merian
