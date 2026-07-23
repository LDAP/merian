#include "merian/utils/camera/camera.hpp"

#include "merian/shader/shader_cursor.hpp"
#include "merian/utils/hash.hpp"
#include "merian/utils/xorshift.hpp"
#include "merian/vk/utils/math.hpp"

#include <array>
#include <cassert>
#include <cmath>

namespace merian {

Camera::Camera(const float3& position,
               const float3& target,
               const float3& up,
               const float field_of_view,
               const float aspect_ratio,
               const float near_plane,
               const float far_plane)
    : position(position), target(target), up(normalize(up)), field_of_view(field_of_view),
      aspect_ratio(aspect_ratio), near_plane(near_plane), far_plane(far_plane) {

    assert(field_of_view < radians(179.99f));
    assert(field_of_view > radians(0.01f));
    assert(near_plane > 0 && near_plane < far_plane);
    assert(far_plane > 0 && far_plane > near_plane);

    view_change_id++;
    projection_change_id++;
}

// -----------------------------------------------------------------------------

const float4x4& Camera::get_view_matrix() noexcept {
    if (has_changed(view_change_id, view_change_id_cache)) {
        view_cache = merian::look_at(position, target, up);
    }
    return view_cache;
}

const float4x4& Camera::get_projection_matrix() noexcept {
    if (has_changed(projection_change_id, projection_change_id_cache)) {
        projection_cache = merian::perspective(field_of_view, aspect_ratio, near_plane, far_plane);
    }
    return projection_cache;
}

float4x4 Camera::get_view_projection_matrix() noexcept {
    return mul(get_projection_matrix(), get_view_matrix());
}

// -----------------------------------------------------------------------------

bool Camera::has_changed_update(uint64_t& check_id) const noexcept {
    return has_changed(get_change_id(), check_id);
}

uint64_t Camera::get_change_id() const noexcept {
    return (uint64_t)view_change_id << 32 | (uint64_t)projection_change_id;
}
bool Camera::operator==(const Camera& other) const noexcept {
    return position == other.position && target == other.target && up == other.up &&
           field_of_view == other.field_of_view && aspect_ratio == other.aspect_ratio &&
           near_plane == other.near_plane && far_plane == other.far_plane &&
           f_number == other.f_number && focus_distance == other.focus_distance &&
           sensor_height == other.sensor_height;
}

// -----------------------------------------------------------------------------

void Camera::look_at(const float3& position, const float3& target, const float3& up) noexcept {
    this->position = position;
    this->target = target;
    this->up = normalize(up);
    view_change_id++;
}

void Camera::look_at(const float3& position,
                     const float3& target,
                     const float3& up,
                     const float field_of_view) noexcept {
    this->position = position;
    this->target = target;
    this->up = normalize(up);
    this->field_of_view = field_of_view;
    view_change_id++;
    projection_change_id++;
}

void Camera::set_position(const float3& position) noexcept {
    this->position = position;
    view_change_id++;
}

void Camera::set_target(const float3& center) noexcept {
    this->target = center;
    view_change_id++;
}

void Camera::set_up(const float3& up) noexcept {
    this->up = normalize(up);
    view_change_id++;
}

const float3& Camera::get_position() const noexcept {
    return position;
}

const float3& Camera::get_target() const noexcept {
    return target;
}

const float3& Camera::get_up() const noexcept {
    return up;
}

// -----------------------------------------------------------------------------

void Camera::set_perspective(const float field_of_view,
                             const float aspect_ratio,
                             const float near_plane,
                             const float far_plane) noexcept {
    assert(field_of_view < radians(179.99f));
    assert(field_of_view > radians(0.01f));
    assert(near_plane > 0 && near_plane < far_plane);
    assert(far_plane > 0 && far_plane > near_plane);

    this->field_of_view = field_of_view;
    this->aspect_ratio = aspect_ratio;
    this->near_plane = near_plane;
    this->far_plane = far_plane;

    projection_change_id++;
}

void Camera::set_field_of_view_vertical(const float field_of_view) noexcept {
    assert(field_of_view < radians(179.99f));
    assert(field_of_view > radians(0.01f));

    this->field_of_view = field_of_view;
    projection_change_id++;
}

void Camera::set_field_of_view_horizontal(const float field_of_view) noexcept {
    assert(field_of_view < radians(179.99f));
    assert(field_of_view > radians(0.01f));

    this->field_of_view = 2.f * std::atan(std::tan(field_of_view * 0.5f) / aspect_ratio);
    projection_change_id++;
}

// aspect_ratio = width / height
void Camera::set_aspect_ratio(const float aspect_ratio) noexcept {
    assert(aspect_ratio > 0);

    this->aspect_ratio = aspect_ratio;
    projection_change_id++;
}

void Camera::set_near_plane(const float near_plane) noexcept {
    assert(near_plane > 0 && near_plane < far_plane);

    this->near_plane = near_plane;
    projection_change_id++;
}

void Camera::set_far_plane(const float far_plane) noexcept {
    assert(far_plane > 0 && far_plane > near_plane);

    this->far_plane = far_plane;
    projection_change_id++;
}

void Camera::set_f_number(const float f_number) noexcept {
    assert(f_number >= 0);

    this->f_number = f_number;
    projection_change_id++;
}

void Camera::set_focus_distance(const float focus_distance) noexcept {
    assert(focus_distance > 0);

    this->focus_distance = focus_distance;
    projection_change_id++;
}

void Camera::set_focal_length(const float focal_length_mm) noexcept {
    assert(focal_length_mm > 0.f);

    field_of_view = 2.f * std::atan((sensor_height * 0.5f) / focal_length_mm);
    projection_change_id++;
}

void Camera::set_sensor_height(const float sensor_height_mm) noexcept {
    assert(sensor_height_mm > 0.f);

    this->sensor_height = sensor_height_mm;
    projection_change_id++;
}

float Camera::get_field_of_view_vertical() const noexcept {
    return field_of_view;
}

float Camera::get_field_of_view_horizontal() const noexcept {
    return 2.f * std::atan(std::tan(field_of_view * 0.5f) * aspect_ratio);
}

float Camera::get_aspect_ratio() const noexcept {
    return aspect_ratio;
}

float Camera::get_near_plane() const noexcept {
    return near_plane;
}

float Camera::get_far_plane() const noexcept {
    return far_plane;
}

float Camera::get_f_number() const noexcept {
    return f_number;
}

float Camera::get_focal_length() const noexcept {
    // inverse of vfov = 2 * atan((sensor_height / 2) / focal_length)
    return (sensor_height * 0.5f) / std::tan(field_of_view * 0.5f);
}

float Camera::get_sensor_height() const noexcept {
    return sensor_height;
}

float Camera::get_aperture_radius() const noexcept {
    if (f_number <= 0.f) {
        return 0.f;
    }
    // aperture diameter = focal length / N; mm -> scene units (1 unit = 1 meter).
    const float focal_length_m = get_focal_length() * 0.001f;
    return focal_length_m / (2.f * f_number);
}

float Camera::get_focus_distance() noexcept {
    if (focus_distance_on_target) {
        focus_distance = merian::length(target - position);
    }
    return focus_distance;
}

void Camera::recompute_ray_basis() {
    forward_cache = normalize(target - position);
    right_cache = normalize(cross(forward_cache, up));
}

const float3& Camera::get_forward() noexcept {
    if (has_changed(get_change_id(), ray_basis_change_id_cache)) {
        recompute_ray_basis();
    }
    return forward_cache;
}

const float3& Camera::get_right() noexcept {
    if (has_changed(get_change_id(), ray_basis_change_id_cache)) {
        recompute_ray_basis();
    }
    return right_cache;
}

void Camera::set_jitter(const float2& jitter) noexcept {
    this->jitter = jitter;
}

namespace {

const std::array<float2, 8> HALTON_SEQUENCE = {
    float2(1.0f / 2.0f, 1.0f / 3.0f), float2(1.0f / 4.0f, 2.0f / 3.0f),
    float2(3.0f / 4.0f, 1.0f / 9.0f), float2(1.0f / 8.0f, 4.0f / 9.0f),
    float2(5.0f / 8.0f, 7.0f / 9.0f), float2(3.0f / 8.0f, 2.0f / 9.0f),
    float2(7.0f / 8.0f, 5.0f / 9.0f), float2(1.0f / 16.0f, 8.0f / 9.0f),
};

// length 16 is a Padovan number -> maximally isotropic plastic-constant (R2) lattice
constexpr uint32_t R2_SEQUENCE_LENGTH = 16;
constexpr float R2_ALPHA_X = 0.7548776662466927f; // 1 / g
constexpr float R2_ALPHA_Y = 0.5698402909980532f; // 1 / g^2

// importance sample the Blackman-Harris pixel filter with 1.5px radius support.
float2 pixel_offset_blackman_harris(const float2& rand) {
    constexpr float two_pi = 6.28318530717958647692f;
    const float2 dir = float2(std::cos(rand.y * two_pi), std::sin(rand.y * two_pi));
    // surprisingly good fit to the inverse cdf.
    const float r = 0.943404f * std::asin(0.636617f * std::asin(std::sqrt(rand.x)));
    return dir * r;
}

} // namespace

void Camera::set_jitter_sequence(const JitterSequence sequence) noexcept {
    jitter_sequence = sequence;
    if (sequence == JitterSequence::None) {
        jitter = float2(0.0f);
    }
}

void Camera::advance_jitter(const uint32_t frame_index) noexcept {
    switch (jitter_sequence) {
    case JitterSequence::None:
        return;
    case JitterSequence::Halton:
        jitter = HALTON_SEQUENCE[frame_index % HALTON_SEQUENCE.size()] - 0.5f;
        break;
    case JitterSequence::R2: {
        const auto n = static_cast<float>(frame_index % R2_SEQUENCE_LENGTH);
        jitter = float2(std::fmod(0.5f + (R2_ALPHA_X * n), 1.0f),
                        std::fmod(0.5f + (R2_ALPHA_Y * n), 1.0f)) -
                 0.5f;
        break;
    }
    case JitterSequence::BlackmanHarris: {
        // random per frame: independent samples of the filter, robust to clearing at any time.
        XORShift32 rng(static_cast<uint32_t>(hash_val(frame_index)));
        jitter = pixel_offset_blackman_harris(float2(rng.next_float(), rng.next_float()));
        break;
    }
    }
}

Camera::JitterSequence Camera::get_jitter_sequence() const noexcept {
    return jitter_sequence;
}

const float2& Camera::get_jitter() const noexcept {
    return jitter;
}

void Camera::write_to(ShaderCursor cursor) {
    const float half_vfov_tan = std::tan(field_of_view * 0.5f);
    const float3& fwd = get_forward();
    const float3& rgt = get_right();
    const float focus = get_focus_distance();

    cursor["position"] = position;
    cursor["target"] = target;
    cursor["up"] = up;
    // basis scaled so position + ndc.x * U + ndc.y * V + W lies on the plane of perfect focus;
    // normalized directions and projections are invariant to the scale.
    cursor["U"] = rgt * (half_vfov_tan * aspect_ratio * focus);
    cursor["V"] = cross(rgt, fwd) * (half_vfov_tan * focus);
    cursor["W"] = fwd * focus;
    cursor["near"] = near_plane;
    cursor["far"] = far_plane;
    cursor["aspect_ratio"] = aspect_ratio;
    cursor["aperture_radius"] = get_aperture_radius();
    cursor["jitter"] = jitter;
}

// High level operations
// -----------------------------------------------------------------------------

void Camera::look_at_bounding_box(const AABB& aabb) {
    assert(aabb.is_valid());

    // reduce field of view to make the fit not tight => looks better
    const float vfov = this->field_of_view * 0.85f;

    const float tan_hy = std::tan(vfov * 0.5f);
    const float tan_hx = tan_hy * aspect_ratio;

    // Establish an initial view direction
    look_at(float3(1.3) * (aabb.get_max().y + 1e-3f), aabb.get_center(), get_up());

    // Iteratively: pan to center the projected AABB, then compute distance to fit.
    // Pan shifts both position and target, preserving the view direction.
    for (int iter = 0; iter < 3; iter++) {
        const float3& fwd = get_forward();
        const float3& r = get_right();
        const float3 u = cross(r, fwd);

        // Project all 8 corners and find the projected bounding rect center
        float px_min = std::numeric_limits<float>::max();
        float px_max = std::numeric_limits<float>::lowest();
        float py_min = std::numeric_limits<float>::max();
        float py_max = std::numeric_limits<float>::lowest();

        for (uint32_t i = 0; i < 8; i++) {
            const float3 v = aabb.get_corner(i) - position;
            const float d = dot(v, fwd);
            const float px = dot(v, r) / d;
            const float py = dot(v, u) / d;
            px_min = std::min(px_min, px);
            px_max = std::max(px_max, px);
            py_min = std::min(py_min, py);
            py_max = std::max(py_max, py);
        }

        const float target_depth = dot(target - position, fwd);
        const float3 shift = ((px_min + px_max) * 0.5f * target_depth) * r +
                             ((py_min + py_max) * 0.5f * target_depth) * u;
        look_at(position + shift, target + shift, up);

        // Compute the distance to fit all corners with the recentered view
        float dist = 0.f;
        for (uint32_t i = 0; i < 8; i++) {
            const float3 v = aabb.get_corner(i) - target;
            const float depth = dot(v, fwd);
            const float dx = (std::abs(dot(v, r)) / tan_hx) - depth;
            const float dy = (std::abs(dot(v, u)) / tan_hy) - depth;
            dist = std::max({dist, dx, dy});
        }

        set_position(target - fwd * dist);
    }
}

void Camera::move(const float dx, const float dup, const float dz) {
    position += dup * up;
    target += dup * up;

    float3 z = position - target;
    if (merian::length(z) < 1e-5)
        return;
    z = normalize(z);

    const float3 x = normalize(cross(up, z));
    const float3 in = normalize(cross(x, up));

    const float3 d = dx * x + dz * in;
    position += d;
    target += d;

    view_change_id++;
}

void Camera::fly(const float dx, const float dy, const float dz) {
    float3 z = position - target;
    if (merian::length(z) < 1e-5)
        return;
    z = normalize(z);

    const float3 x = normalize(cross(up, z));
    const float3 y = normalize(cross(z, x));

    const float3 d = dx * x + dy * y + dz * z;
    position += d;
    target += d;

    view_change_id++;
}

void Camera::rotate(const float d_phi, const float d_theta) {
    rotate_around(target, position, up, d_phi, d_theta);
    view_change_id++;
}

// Orbit around the "center" horizontally (phi) or vertically (theta).
//  * pi equals a full turn.
void Camera::orbit(const float d_phi, const float d_theta) {
    rotate_around(position, target, up, d_phi, d_theta);
    view_change_id++;
}

void Camera::dolly(const float distance) {
    float3 z = position - target;
    const float len = merian::length(z);
    if (len < 1e-5f)
        return;
    z = z / len;
    // Clamp toward the target so the eye never reaches or crosses it.
    position += z * std::max(distance, -len + 1e-3f);
    view_change_id++;
}

void Camera::properties(Properties& props) {
    if (props.config_vec("position", position)) {
        view_change_id++;
    }
    if (props.config_vec("target", target)) {
        view_change_id++;
    }
    if (props.config_vec("up", up)) {
        view_change_id++;
    }

    props.st_separate();

    float fov_hdeg = degrees(get_field_of_view_horizontal());
    if (props.config_float("fov", fov_hdeg, "horizontal, in degrees", 0.1f, 0.01f, 179.9f)) {
        set_field_of_view_horizontal(radians(fov_hdeg));
    }
    float focal_length = get_focal_length();
    if (props.config_float("focal length", focal_length,
                           "in mm, derived from vertical FOV and sensor height", 0.1f, 1.0f,
                           2000.0f)) {
        set_focal_length(focal_length);
    }
    if (props.config_float("near", near_plane, "", 0.1f, 0.0f, far_plane - 0.1f)) {
        projection_change_id++;
    }
    if (props.config_float("far", far_plane, "", 0.1f, near_plane + 0.1f, 10000.0f)) {
        projection_change_id++;
    }
    if (props.config_float("aspect", aspect_ratio)) {
        projection_change_id++;
    }

    props.st_separate();

    static const std::vector<std::string> sensor_presets = {"Full-frame", "APS-C",   "Micro 4/3",
                                                            "1\"",        "1/2.3\"", "Custom"};
    static const std::array<float, 5> sensor_heights = {24.0f, 15.6f, 13.0f, 8.8f, 4.55f};
    int sensor_selected = static_cast<int>(sensor_heights.size()); // Custom
    for (size_t i = 0; i < sensor_heights.size(); i++) {
        if (std::abs(sensor_height - sensor_heights[i]) < 1e-3f) {
            sensor_selected = static_cast<int>(i);
            break;
        }
    }
    if (props.config_options("sensor", sensor_selected, sensor_presets,
                             Properties::OptionsStyle::COMBO) &&
        sensor_selected < static_cast<int>(sensor_heights.size())) {
        set_sensor_height(sensor_heights[sensor_selected]);
    }
    if (props.config_float("sensor height", sensor_height, "image height in mm (vertical)", 0.01f,
                           0.1f, 100.0f)) {
        set_sensor_height(sensor_height);
    }
    if (props.config_float("f-stop", f_number,
                           "f-number of the lens, 0 = pinhole; assumes 1 scene unit = 1 meter",
                           0.1f, 0.0f, 32.0f)) {
        projection_change_id++;
    }
    if (props.config_float("focus distance", focus_distance,
                           "distance to the plane of perfect focus", 0.001f, 0.01f, 20.0f)) {
        focus_distance_on_target = false;
        projection_change_id++;
    }
    if (props.config_bool("focus on target", focus_distance_on_target,
                          "lock the focus distance to the distance to the target") &&
        focus_distance_on_target) {
        focus_distance = merian::length(target - position);
        projection_change_id++;
    }

    if (props.is_ui()) {
        props.st_separate();

        float3 d_fly(0);
        if (props.config_vec("fly", d_fly)) {
            fly(d_fly.x, d_fly.y, d_fly.z);
        }
        float3 d_move(0);
        if (props.config_vec("move", d_move)) {
            move(d_move.x, d_move.y, d_move.z);
        }
        float2 d_rotate(0);
        if (props.config_vec("rotate", d_rotate)) {
            rotate(d_rotate.x, d_rotate.y);
        }
        float2 d_orbit(0);
        if (props.config_vec("orbit", d_orbit)) {
            orbit(d_orbit.x, d_orbit.y);
        }
    }

    props.st_separate();

    static const std::vector<std::string> jitter_sequences = {"None", "Halton", "R2",
                                                              "Blackman-Harris"};
    int selected = static_cast<int>(jitter_sequence);
    if (props.config_options("jitter", selected, jitter_sequences,
                             Properties::OptionsStyle::COMBO)) {
        set_jitter_sequence(static_cast<JitterSequence>(selected));
    }
}

} // namespace merian
