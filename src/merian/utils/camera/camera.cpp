#include "merian/utils/camera/camera.hpp"

namespace merian {

Camera::Camera(const glm::vec3& eye,
               const glm::vec3& center,
               const glm::vec3& up,
               const float field_of_view,
               const float aspect_ratio,
               const float near_plane,
               const float far_plane)
    : eye(eye), center(center), up(glm::normalize(up)), field_of_view(field_of_view),
      aspect_ratio(aspect_ratio), near_plane(near_plane), far_plane(far_plane) {

    assert(field_of_view < 179.99);
    assert(field_of_view > 0.01);
    assert(near_plane > 0 && near_plane < far_plane);
    assert(far_plane > 0 && far_plane > near_plane);

    view_change_id++;
    projection_change_id++;
}

// -----------------------------------------------------------------------------

const glm::mat4& Camera::get_view_matrix() noexcept {
    if (has_changed(view_change_id, view_change_id_cache)) {
        view_cache = glm::lookAt(eye, center, up);
    }
    return view_cache;
}

const glm::mat4& Camera::get_projection_matrix() noexcept {
    if (has_changed(projection_change_id, projection_change_id_cache)) {
        projection_cache = glm::perspective(field_of_view, aspect_ratio, near_plane, far_plane);
    }
    return projection_cache;
}

glm::mat4 Camera::get_view_projection_matrix() noexcept {
    return get_projection_matrix() * get_view_matrix();
}

// -----------------------------------------------------------------------------

bool Camera::has_changed_update(uint64_t& check_id) const noexcept {
    return has_changed(get_change_id(), check_id);
}

uint64_t Camera::get_change_id() const noexcept {
    return (uint64_t)view_change_id << 32 | (uint64_t)projection_change_id;
}

// -----------------------------------------------------------------------------

void Camera::look_at(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up) noexcept {
    this->eye = eye;
    this->center = center;
    this->up = glm::normalize(up);
    view_change_id++;
}

void Camera::look_at(const glm::vec3& eye,
                     const glm::vec3& center,
                     const glm::vec3& up,
                     const float field_of_view) noexcept {
    this->eye = eye;
    this->center = center;
    this->up = glm::normalize(up);
    this->field_of_view = field_of_view;
    view_change_id++;
    projection_change_id++;
}

void Camera::set_eye(const glm::vec3& eye) noexcept {
    this->eye = eye;
    view_change_id++;
}

void Camera::set_center(const glm::vec3& center) noexcept {
    this->center = center;
    view_change_id++;
}

void Camera::set_up(const glm::vec3& up) noexcept {
    this->up = glm::normalize(up);
    view_change_id++;
}

const glm::vec3& Camera::get_eye() const noexcept {
    return eye;
}

const glm::vec3& Camera::get_center() const noexcept {
    return center;
}

const glm::vec3& Camera::get_up() const noexcept {
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

void Camera::look_at_bounding_box(const glm::vec3& box_min, const glm::vec3& box_max, bool tight) {
    const glm::vec3 bb_half_dimensions = (box_max - box_min) * .5f;
    const glm::vec3 bb_center = box_min + bb_half_dimensions;

    float offset = 0;
    float yfov = field_of_view;
    float xfov = field_of_view * aspect_ratio;

    if (!tight) {
        // Using the bounding sphere
        float radius = glm::length(bb_half_dimensions);
        if (aspect_ratio > 1.f)
            offset = radius / sin(glm::radians(yfov * 0.5f));
        else
            offset = radius / sin(glm::radians(xfov * 0.5f));
    } else {
        // keep only rotation
        glm::mat3 mView = glm::lookAt(eye, bb_center, up);

        for (int i = 0; i < 8; i++) {
            glm::vec3 vct(i & 1 ? bb_half_dimensions.x : -bb_half_dimensions.x,
                          i & 2 ? bb_half_dimensions.y : -bb_half_dimensions.y,
                          i & 4 ? bb_half_dimensions.z : -bb_half_dimensions.z);
            vct = mView * vct;

            if (vct.z < 0) // Take only points in front of the center
            {
                // Keep the largest offset to see that vertex
                offset = std::max(glm::abs(vct.y) / glm::tan(glm::radians(yfov * 0.5f)) +
                                      glm::abs(vct.z),
                                  offset);
                offset = std::max(glm::abs(vct.x) / glm::tan(glm::radians(xfov * 0.5f)) +
                                      glm::abs(vct.z),
                                  offset);
            }
        }
    }

    auto view_direction = glm::normalize(eye - center);
    auto new_eye = bb_center + view_direction * offset;

    // updates all matrices and change id
    look_at(new_eye, bb_center, up);
}

void Camera::move(const float dx, const float dup, const float dz) {
    eye += dup * up;
    center += dup * up;

    glm::vec3 z = eye - center;
    if (glm::length(z) < 1e-5)
        return;
    z = glm::normalize(z);

    const glm::vec3 x = glm::normalize(glm::cross(up, z));
    const glm::vec3 in = glm::normalize(glm::cross(x, up));

    const glm::vec3 d = dx * x + dz * in;
    eye += d;
    center += d;

    view_change_id++;
}

void Camera::fly(const float dx, const float dy, const float dz) {
    glm::vec3 z = eye - center;
    if (glm::length(z) < 1e-5)
        return;
    z = glm::normalize(z);

    const glm::vec3 x = glm::normalize(glm::cross(up, z));
    const glm::vec3 y = glm::normalize(glm::cross(z, x));

    const glm::vec3 d = dx * x + dy * y + dz * z;
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
