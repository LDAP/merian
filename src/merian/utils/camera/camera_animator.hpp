#pragma once

#include "merian/utils/camera/camera.hpp"
#include "merian/utils/interpolation.hpp"

#include <chrono>

namespace merian {

/**
 * @brief      An animator for the camera.
 *
 * Provides smooth camera motion and animation.
 *
 * The animator does not update internally (no thread is started) instead the user must
 * call one of the update() methods periodically.
 */
class CameraAnimator {
  private:
    using chrono_clock = std::chrono::high_resolution_clock;

  public:
    CameraAnimator(double animation_duration_ms = 0.5);

    void update(const chrono_clock::time_point now);

    void set_camera_target(const Camera& camera, bool animate = true);

    // The animated camera
    const Camera& get_current_camera();

    // The camera that is pursued by the animator
    const Camera& get_camera_target();

    bool is_animating();

  private:
    // calculates the bezier points for a smooth animation between start end end.
    void calculate_eye_animation_bezier_points();

  private:
    Camera camera_current;

    Camera animation_start;
    Camera animation_end;
    // Animate eye using bezier for consistent animation
    glm::mat3 eye_animation_bezier_points;
    std::optional<chrono_clock::time_point> animation_start_time;

    double animation_duration_ms;
};

} // namespace merian
