#pragma once

#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include <glm/glm.hpp>

namespace merian {

class Camera {
  private:
    // Checks if current_id != check_id and sets check_id = current_id
    template <typename T> static bool has_changed(const T current_id, T& check_id) {
        if (check_id == current_id) {
            return false;
        } else {
            check_id = current_id;
            return true;
        }
    }

  public:
    Camera(const glm::vec3& eye = glm::vec3(0),
           const glm::vec3& center = glm::vec3(1, 0, 0),
           const glm::vec3& up = glm::vec3(0, 0, 1),
           const float field_of_view = 60.f,
           const float aspect_ratio = 1.f,
           const float near_plane = 0.1f,
           const float far_plane = 1000.f)
        : eye(eye), center(center), up(up), field_of_view(field_of_view), aspect_ratio(aspect_ratio),
          near_plane(near_plane), far_plane(far_plane) {

        assert(field_of_view < 179.99);
        assert(field_of_view > 0.01);
        assert(near_plane > 0 && near_plane < far_plane);
        assert(far_plane > 0 && far_plane > near_plane);

        view_change_id++;
        projection_change_id++;
    }

    bool operator==(const Camera&) const = default;

    // -----------------------------------------------------------------------------

    const glm::mat4& get_view_matrix() noexcept {
        if (has_changed(view_change_id, view_change_id_cache)) {
            view_cache = glm::lookAt(eye, center, up);
        }
        return view_cache;
    }

    const glm::mat4& get_projection_matrix() noexcept {
        if (has_changed(projection_change_id, projection_change_id_cache)) {
            projection_cache = glm::perspective(field_of_view, aspect_ratio, near_plane, far_plane);
        }
        return projection_cache;
    }

    glm::mat4 get_view_projection_matrix() noexcept {
        return get_projection_matrix() * get_view_matrix();
    }

    // -----------------------------------------------------------------------------

    // Convenience method that checks if the camera changed
    // and updates the supplied ID to the current ID.
    bool has_changed_update(uint64_t& check_id) noexcept {
        return has_changed(get_change_id(), check_id);
    }

    uint64_t get_change_id() noexcept {
        return (uint64_t)view_change_id << 32 | (uint64_t)projection_change_id;
    }

    // -----------------------------------------------------------------------------

    void look_at(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up) noexcept {
        this->eye = eye;
        this->center = center;
        this->up = up;
        view_change_id++;
    }

    void look_at(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up, const float field_of_view) noexcept {
        this->eye = eye;
        this->center = center;
        this->up = up;
        this->field_of_view = field_of_view;
        view_change_id++;
        projection_change_id++;
    }

    void set_eye(const glm::vec3& eye) noexcept {
        this->eye = eye;
        view_change_id++;
    }

    void set_center(const glm::vec3& center) noexcept {
        this->center = center;
        view_change_id++;
    }

    void set_up(const glm::vec3& up) noexcept {
        this->up = up;
        view_change_id++;
    }

    const glm::vec3& get_eye() const noexcept {
        return eye;
    }

    const glm::vec3& get_center() const noexcept {
        return center;
    }

    const glm::vec3& get_up() const noexcept {
        return up;
    }

    // -----------------------------------------------------------------------------

    void set_perspective(const float field_of_view = 60.f,
                         const float aspect_ratio = 1.f,
                         const float near_plane = 0.1f,
                         const float far_plane = 1000.f) noexcept {
        assert(field_of_view < 179.99);
        assert(field_of_view > 0.01);
        assert(near_plane > 0 && near_plane < far_plane);
        assert(far_plane > 0 && far_plane > near_plane);

        this->field_of_view = field_of_view;
        this->aspect_ratio = aspect_ratio;
        this->near_plane = near_plane;
        this->far_plane = far_plane;
    }

    void set_field_of_view(const float field_of_view) noexcept {
        assert(field_of_view < 179.99);
        assert(field_of_view > 0.01);

        this->field_of_view = field_of_view;
        projection_change_id++;
    }

    // aspect_ratio = width / height
    void set_aspect_ratio(const float aspect_ratio) noexcept {
        assert(aspect_ratio > 0);

        this->aspect_ratio = aspect_ratio;
        projection_change_id++;
    }

    void set_near_plane(const float near_plane) noexcept {
        assert(near_plane > 0 && near_plane < far_plane);

        this->near_plane = near_plane;
        projection_change_id++;
    }

    void set_far_plane(const float far_plane) noexcept {
        assert(far_plane > 0 && far_plane > near_plane);

        this->far_plane = far_plane;
        projection_change_id++;
    }

    float get_field_of_view() const noexcept {
        return field_of_view;
    }

    // High level operations
    // -----------------------------------------------------------------------------

    // Fitting the camera position and interest to see the bounding box
    // tight: Fit bounding box exactly, not tight: fit bounding sphere
    void look_at_bounding_box(const glm::vec3& box_min, const glm::vec3& box_max, bool tight = false) {
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
                    offset = std::max(glm::abs(vct.y) / glm::tan(glm::radians(yfov * 0.5f)) + glm::abs(vct.z), offset);
                    offset = std::max(glm::abs(vct.x) / glm::tan(glm::radians(xfov * 0.5f)) + glm::abs(vct.z), offset);
                }
            }
        }

        auto view_direction = glm::normalize(eye - center);
        auto new_eye = bb_center + view_direction * offset;

        // updates all matrices and change id
        look_at(new_eye, bb_center, up);
    }

  private:
    // VIEW
    //-------------------------------------------------

    glm::vec3 eye;    // Position of the camera
    glm::vec3 center; // Position where the camera is looking at
    glm::vec3 up;     // Normalized up vector where the camera is oriented

    // Increase whenever eye, center or up changes
    uint32_t view_change_id = 0;

    // Cache

    // Do not use diectly
    glm::mat4 view_cache;
    uint32_t view_change_id_cache = 0;

    // PROJECTION
    //-------------------------------------------------

    float field_of_view; // in degree
    float aspect_ratio;  // aspect_ratio = width / height
    float near_plane;
    float far_plane;

    // Increase whenever fov, aspect_ratio, near_plane or far_plane changes
    uint32_t projection_change_id = 0;

    // Cache

    // Do not use diectly
    glm::mat4 projection_cache;
    uint32_t projection_change_id_cache = 0;
};

} // namespace merian
