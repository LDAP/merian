#include "merian/utils/camera/camera_animator.hpp"

#include "merian/utils/interpolation.hpp"

#include <algorithm>
#include <cmath>

namespace merian {

namespace {
// Positions compared relative to the eye-target distance, directions/fov absolutely.
bool settled(const Camera& a, const Camera& b) {
    const float scale = std::max(merian::length(a.get_position() - a.get_target()), 1e-3f);
    return merian::length(a.get_position() - b.get_position()) < 1e-4f * scale &&
           merian::length(a.get_target() - b.get_target()) < 1e-4f * scale &&
           merian::length(a.get_up() - b.get_up()) < 1e-5f &&
           std::abs(a.get_field_of_view_vertical() - b.get_field_of_view_vertical()) < 1e-6f;
}
} // namespace

void CameraAnimator::follow(const Camera& target, const double dt_seconds, const double tau_s) {
    if (settled(camera_current, target)) {
        camera_current = target;
        return;
    }
    const float a = static_cast<float>(1.0 - std::exp(-dt_seconds / std::max(tau_s, 1e-6)));
    camera_current.look_at(
        lerp(camera_current.get_position(), target.get_position(), a),
        lerp(camera_current.get_target(), target.get_target(), a),
        lerp(camera_current.get_up(), target.get_up(), a),
        lerp(camera_current.get_field_of_view_vertical(), target.get_field_of_view_vertical(), a));
}

} // namespace merian
