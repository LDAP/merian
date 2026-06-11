#pragma once

#include "merian/utils/camera/camera_animator.hpp"
#include "merian/utils/input_listener.hpp"

namespace merian {

class Properties;

// Drives an attached camera through merian's InputController. Blender-style bindings:
//   - hold left/middle mouse: orbit; Shift+drag pans; scroll dollies toward the target (smoothed).
//   - hold right mouse: fly with WASD + Space/C + mouse-look; Shift boosts; scroll sets speed.
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

    // Re-syncs to a freshly attached camera, then keeps driving it on update().
    void attach(const CameraHandle& camera);

    // Folds accumulated input into the attached camera: fly instantly, orbit eased.
    void update(double dt_seconds);

    void properties(Properties& config);

  private:
    bool is_smoothed() const {
        return left_held || middle_held;
    }

    CameraHandle attached;
    Camera target_camera;
    CameraAnimator animator;

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
};

} // namespace merian
