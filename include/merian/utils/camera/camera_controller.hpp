#pragma once

#include "merian/utils/camera/camera.hpp"
#include "merian/utils/input_listener.hpp"

namespace merian {

class Properties;

// Drives an attached camera through merian's InputController. Blender-style bindings:
//   - hold left/middle mouse: orbit; Shift+drag pans; scroll dollies toward the target (smoothed).
//   - hold right mouse: fly with WASD + Space/C + mouse-look; Shift boosts; scroll sets speed.
//
// Input is accumulated as deltas and applied relative to the camera's current state, so external
// changes (e.g. the camera's properties UI) compose with navigation instead of being overwritten.
class CameraController : public InputListener {
  public:
    bool on_key(InputController& controller,
                InputController::Key key,
                InputController::KeyStatus action,
                int mods) override;

    bool on_mouse_button(InputController& controller,
                         InputController::MouseButton button,
                         InputController::KeyStatus status) override;

    bool on_cursor(InputController& controller, double xpos, double ypos) override;

    bool on_scroll(InputController& controller, double xoffset, double yoffset) override;

    void attach(const CameraHandle& camera);

    // Applies pending input to the attached camera: fly/look instantly, orbit/pan/dolly eased.
    void update(double dt_seconds);

    void properties(Properties& config);

  private:
    CameraHandle attached;

    // Pending input deltas, consumed by update().
    float2 pending_look{0};  // fly mouse-look (phi, theta), radians
    float2 pending_orbit{0}; // orbit (phi, theta), radians
    float2 pending_pan{0};   // cursor drag, pixels
    float pending_dolly = 0; // fraction of the eye-target distance

    bool key_forward = false;
    bool key_back = false;
    bool key_left = false;
    bool key_right = false;
    bool key_up = false;
    bool key_down = false;
    bool key_fast = false;
    bool key_pan = false;

    bool right_held = false;
    bool left_held = false;
    bool middle_held = false;
    bool have_cursor = false;
    double last_x = 0.0;
    double last_y = 0.0;

    float move_speed = 3.0f;
    float look_sensitivity = 0.0015f;
    float orbit_sensitivity = 0.01f;
    float pan_sensitivity = 0.002f;
    float dolly_fraction = 0.1f;
    float smoothing_tau = 0.03f; // exponential easing time constant, seconds
};

} // namespace merian
