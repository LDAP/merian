#include "merian/utils/camera/camera.hpp"

#include "merian/vk/utils/math.hpp"

#include <cassert>

namespace merian {

Camera::Camera(const float3& eye,
               const float3& center,
               const float3& up,
               const float field_of_view,
               const float aspect_ratio,
               const float near_plane,
               const float far_plane)
    : eye(eye), center(center), up(normalize(up)), field_of_view(field_of_view),
      aspect_ratio(aspect_ratio), near_plane(near_plane), far_plane(far_plane) {

    assert(field_of_view < 179.99);
    assert(field_of_view > 0.01);
    assert(near_plane > 0 && near_plane < far_plane);
    assert(far_plane > 0 && far_plane > near_plane);

    view_change_id++;
    projection_change_id++;
}

// -----------------------------------------------------------------------------

const float4x4& Camera::get_view_matrix() noexcept {
    if (has_changed(view_change_id, view_change_id_cache)) {
        view_cache = float4x4::look_at(eye, center, up);
    }
    return view_cache;
}

const float4x4& Camera::get_projection_matrix() noexcept {
    // if (has_changed(projection_change_id, projection_change_id_cache)) {
    //     projection_cache = float4x4::perspective(projection(frustum()))//
    //     glm::perspective(field_of_view, aspect_ratio, near_plane, far_plane);
    // }
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

// -----------------------------------------------------------------------------

void Camera::look_at(const float3& eye, const float3& center, const float3& up) noexcept {
    this->eye = eye;
    this->center = center;
    this->up = normalize(up);
    view_change_id++;
}

void Camera::look_at(const float3& eye,
                     const float3& center,
                     const float3& up,
                     const float field_of_view) noexcept {
    this->eye = eye;
    this->center = center;
    this->up = normalize(up);
    this->field_of_view = field_of_view;
    view_change_id++;
    projection_change_id++;
}

void Camera::set_eye(const float3& eye) noexcept {
    this->eye = eye;
    view_change_id++;
}

void Camera::set_center(const float3& center) noexcept {
    this->center = center;
    view_change_id++;
}

void Camera::set_up(const float3& up) noexcept {
    this->up = normalize(up);
    view_change_id++;
}

const float3& Camera::get_eye() const noexcept {
    return eye;
}

const float3& Camera::get_center() const noexcept {
    return center;
}

const float3& Camera::get_up() const noexcept {
    return up;
}

// -----------------------------------------------------------------------------

void Camera::set_perspective(const float field_of_view,
                             const float aspect_ratio,
                             const float near_plane,
                             const float far_plane) noexcept {
    assert(field_of_view < 179.99);
    assert(field_of_view > 0.01);
    assert(near_plane > 0 && near_plane < far_plane);
    assert(far_plane > 0 && far_plane > near_plane);

    this->field_of_view = field_of_view;
    this->aspect_ratio = aspect_ratio;
    this->near_plane = near_plane;
    this->far_plane = far_plane;
}

void Camera::set_field_of_view(const float field_of_view) noexcept {
    assert(field_of_view < 179.99);
    assert(field_of_view > 0.01);

    this->field_of_view = field_of_view;
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

float Camera::get_field_of_view() const noexcept {
    return field_of_view;
}

// High level operations
// -----------------------------------------------------------------------------

void Camera::look_at_bounding_box(const float3& box_min, const float3& box_max, bool tight) {
    const float3 bb_half_dimensions = (box_max - box_min) * .5f;
    const float3 bb_center = box_min + bb_half_dimensions;

    float offset = 0;
    float yfov = field_of_view;
    float xfov = field_of_view * aspect_ratio;

    if (!tight) {
        // Using the bounding sphere
        float radius = length(bb_half_dimensions);
        if (aspect_ratio > 1.f)
            offset = radius / sin(merian::radians(yfov * 0.5f));
        else
            offset = radius / sin(merian::radians(xfov * 0.5f));
    } else {
        // keep only rotation
        float4x4 view = float4x4::look_at(eye, bb_center, up);

        for (int i = 0; i < 8; i++) {
            float4 vct(i & 1 ? bb_half_dimensions.x : -bb_half_dimensions.x,
                       i & 2 ? bb_half_dimensions.y : -bb_half_dimensions.y,
                       i & 4 ? bb_half_dimensions.z : -bb_half_dimensions.z, 0.f);
            vct = mul(view, vct);

            if (vct.z < 0) // Take only points in front of the center
            {
                // Keep the largest offset to see that vertex
                offset =
                    std::max(abs(vct.y) / tan(merian::radians(yfov * 0.5f)) + abs(vct.z), offset);
                offset =
                    std::max(abs(vct.x) / tan(merian::radians(xfov * 0.5f)) + abs(vct.z), offset);
            }
        }
    }

    auto view_direction = normalize(eye - center);
    auto new_eye = bb_center + view_direction * offset;

    // updates all matrices and change id
    look_at(new_eye, bb_center, up);
}

void Camera::move(const float dx, const float dup, const float dz) {
    eye += dup * up;
    center += dup * up;

    float3 z = eye - center;
    if (length(z) < float1(1e-5))
        return;
    z = normalize(z);

    const float3 x = normalize(cross(up, z));
    const float3 in = normalize(cross(x, up));

    const float3 d = dx * x + dz * in;
    eye += d;
    center += d;

    view_change_id++;
}

void Camera::fly(const float dx, const float dy, const float dz) {
    float3 z = eye - center;
    if (length(z) < float1(1e-5))
        return;
    z = normalize(z);

    const float3 x = normalize(cross(up, z));
    const float3 y = normalize(cross(z, x));

    const float3 d = dx * x + dy * y + dz * z;
    eye += d;
    center += d;

    view_change_id++;
}

void Camera::rotate(const float d_phi, const float d_theta) {
    rotate_around(center, eye, up, d_phi, d_theta);
    view_change_id++;
}

// Orbit around the "center" horizontally (phi) or vertically (theta).
//  * pi equals a full turn.
void Camera::orbit(const float d_phi, const float d_theta) {
    rotate_around(eye, center, up, d_phi, d_theta);
    view_change_id++;
}

} // namespace merian
