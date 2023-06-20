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

    void update(const chrono_clock::time_point now) {
        if (!is_animating()) {
            return;
        }

        uint64_t elapsed_millis = std::chrono::duration<double, std::milli>(now - animation_start_time.value()).count();
        // in [0, 1]
        float t = elapsed_millis / animation_duration_ms;

        if (t >= 1) {
            // animation is done
            camera_current = animation_end;
            animation_start_time = std::nullopt;
            return;
        }

        double smoothed = smootherstep(t);

        // Interpolate camera start and end position
        // Use bezier to connect the camera positon (eye) for consistent animation
        float interpolated_fov =
            glm::mix(animation_start.get_field_of_view(), animation_end.get_field_of_view(), smoothed);
        glm::vec3 interpolated_center = glm::mix(animation_start.get_center(), animation_end.get_center(), smoothed);
        glm::vec3 interpolated_up = glm::mix(animation_start.get_up(), animation_end.get_up(), smoothed);
        glm::vec3 interpolated_eye = evaluate_bezier(smoothed, eye_animation_bezier_points[0],
                                                     eye_animation_bezier_points[1], eye_animation_bezier_points[2]);
        camera_current.look_at(interpolated_eye, interpolated_center, interpolated_up, interpolated_fov);
    }

    void set_camera_target(const Camera& camera, bool animate = true) {
        if (animate) {
            animation_start_time = chrono_clock::now();
            animation_start = camera_current;
            animation_end = camera;
        } else {
            animation_start_time = std::nullopt;
            camera_current = camera;
        }
    }

    // The animated camera
    const Camera& get_current_camera() {
        return camera_current;
    }

    // The camera that is pursued by the animator
    const Camera& get_camera_target() {
        return animation_end;
    }

    bool is_animating() {
        return animation_start_time.has_value();
    }

  private:
    // calculates the bezier points for a smooth animation between start end end.
    void calculate_eye_animation_bezier_points() {
        const glm::vec3 p0 = animation_start.get_eye();
        const glm::vec3 p2 = animation_end.get_eye();
        glm::vec3 p1, pc;

        // point of interest
        const glm::vec3 pi = (animation_end.get_center() + animation_start.get_center()) * 0.5f;

        const glm::vec3 p02 = (p0 + p2) * 0.5f;                          // mid p0-p2
        const float radius = (length(p0 - pi) + length(p2 - pi)) * 0.5f; // Radius for p1
        glm::vec3 p02pi(glm::normalize(p02 - pi));                       // Vector from interest to mid point
        p02pi *= radius;
        pc = pi + p02pi;                       // Calculated point to go through
        p1 = 2.f * pc - p0 * 0.5f - p2 * 0.5f; // Computing p1 for t=0.5
        p1.y = p02.y;                          // Clamping the P1 to be in the same height as p0-p2

        eye_animation_bezier_points = glm::mat3(p0, p1, p2);
    }

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
