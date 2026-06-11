#pragma once

#include "merian/utils/camera/camera.hpp"

namespace merian {

// Eases a camera toward a moving target with frame-rate-independent exponential damping, to smooth
// interactive (e.g. orbit) navigation.
class CameraAnimator {
  public:
    // tau_s is the damping time constant; keep it small (a few frames) to stay responsive.
    void follow(const Camera& target, double dt_seconds, double tau_s = 0.03);

    void snap(const Camera& camera) {
        camera_current = camera;
    }

    const Camera& get_current_camera() const {
        return camera_current;
    }

  private:
    Camera camera_current;
};

} // namespace merian
