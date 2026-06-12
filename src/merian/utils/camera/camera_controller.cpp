#include "merian/utils/camera/camera_controller.hpp"

#include "merian/utils/properties.hpp"

#include <algorithm>
#include <cmath>

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
            pending_look += float2(dx * look_sensitivity, -dy * look_sensitivity);
        } else if (key_pan) {
            pending_pan += float2(dx, dy);
        } else {
            // Orbit moves the eye around the target, so the vertical sign is the opposite of look.
            pending_orbit += float2(dx * orbit_sensitivity, dy * orbit_sensitivity);
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
        pending_dolly += static_cast<float>(-yoffset) * dolly_fraction;
    }
    return true;
}

void CameraController::attach(const CameraHandle& camera) {
    if (camera == attached) {
        return;
    }
    attached = camera;
    pending_look = float2(0);
    pending_orbit = float2(0);
    pending_pan = float2(0);
    pending_dolly = 0.f;
}

void CameraController::update(const double dt_seconds) {
    if (!attached) {
        return;
    }

    // fly and mouse-look track the input exactly
    const float right = static_cast<float>(key_right) - static_cast<float>(key_left);
    const float up = static_cast<float>(key_up) - static_cast<float>(key_down);
    const float back = static_cast<float>(key_back) - static_cast<float>(key_forward);
    if (right_held && (right != 0.0f || up != 0.0f || back != 0.0f)) {
        const float3 step = normalize(float3{right, up, back}) *
                            static_cast<float>(move_speed * (key_fast ? 4.0 : 1.0) * dt_seconds);
        attached->fly(step.x, step.y, step.z);
    }
    if (pending_look != float2(0)) {
        attached->rotate(pending_look.x, pending_look.y);
        pending_look = float2(0);
    }

    // Orbit/pan/dolly ease: consuming an exponential fraction of the pending delta per frame
    // equals frame-rate-independent damping toward the input target.
    const float a =
        1.f - static_cast<float>(std::exp(-dt_seconds / std::max(smoothing_tau, 1e-6f)));
    const float keep = 1.f - a;

    if (pending_orbit != float2(0)) {
        attached->orbit(pending_orbit.x * a, pending_orbit.y * a);
        pending_orbit *= keep;
        if (merian::length(pending_orbit) < 1e-5f) {
            pending_orbit = float2(0);
        }
    }
    if (pending_pan != float2(0)) {
        // Scale by the target distance so panning keeps pace at any zoom.
        const float distance = merian::length(attached->get_position() - attached->get_target());
        attached->move(-pending_pan.x * a * pan_sensitivity * distance,
                       pending_pan.y * a * pan_sensitivity * distance, 0.0f);
        pending_pan *= keep;
        if (merian::length(pending_pan) < 1e-2f) {
            pending_pan = float2(0);
        }
    }
    if (pending_dolly != 0.f) {
        const float distance = merian::length(attached->get_position() - attached->get_target());
        attached->dolly(pending_dolly * a * distance);
        pending_dolly *= keep;
        if (std::abs(pending_dolly) < 1e-4f) {
            pending_dolly = 0.f;
        }
    }
}

void CameraController::properties(Properties& config) {
    config.config_float("look sensitivity", look_sensitivity, "fly-mode mouse look (rad/pixel)",
                        0.0001f, 0.0f);
    config.config_float("orbit sensitivity", orbit_sensitivity, "orbit drag (rad/pixel)", 0.0001f,
                        0.0f);
    config.config_float("pan sensitivity", pan_sensitivity, "pan drag (fraction of distance/pixel)",
                        0.0001f, 0.0f);
    config.config_float("move speed", move_speed, "fly WASD speed (units/s)", 0.05f, 0.0f);
    config.config_float("smoothing", smoothing_tau, "orbit/pan/dolly easing time constant (s)",
                        0.001f, 0.0f, 1.0f);
}

} // namespace merian
