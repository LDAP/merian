#pragma once

#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "merian/vk/utils/math.hpp"
#include <glm/glm.hpp>

namespace merian {

/**
 * @brief      This class describes a camera.
 *
 * The local coordinate system is x: right, y: up, and the camera looks into -z.
 */
class Camera {
  private:
    // Checks if current_id != check_id and sets check_id = current_id
    template <typename T> static bool has_changed(const T current_id, T& check_id) {
        if (check_id == current_id) {
            return false;
        }

        check_id = current_id;
        return true;
    }

  public:
    /**
     * @param[in]  position       The position of the camera
     * @param[in]  target         The position the camera is looking at
     * @param[in]  up             The worlds upward direction
     * @param[in]  field_of_view  The field of view
     * @param[in]  aspect_ratio   The aspect ratio of the camera
     * @param[in]  near_plane     The near plane
     * @param[in]  far_plane      The far plane
     */
    Camera(const glm::vec3& position = glm::vec3(0),
           const glm::vec3& target = glm::vec3(1, 0, 0),
           const glm::vec3& up = glm::vec3(0, 0, 1),
           const float field_of_view = 60.f,
           const float aspect_ratio = 1.f,
           const float near_plane = 0.1f,
           const float far_plane = 1000.f);

    bool operator==(const Camera&) const = default;

    // -----------------------------------------------------------------------------

    const glm::mat4& get_view_matrix() noexcept;

    const glm::mat4& get_projection_matrix() noexcept;

    glm::mat4 get_view_projection_matrix() noexcept;

    const glm::vec3& get_position() const noexcept;

    const glm::vec3& get_target() const noexcept;

    const glm::vec3& get_up() const noexcept;

    glm::vec3 get_forward() const noexcept;

    // -----------------------------------------------------------------------------

    // Convenience method that checks if the camera changed
    // and updates the supplied ID to the current ID.
    bool has_changed_update(uint64_t& check_id) const noexcept;

    uint64_t get_change_id() const noexcept;

    // -----------------------------------------------------------------------------

    void look_at(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) noexcept;

    void look_at(const glm::vec3& position,
                 const glm::vec3& target,
                 const glm::vec3& up,
                 const float field_of_view) noexcept;

    void set_position(const glm::vec3& position) noexcept;

    void set_target(const glm::vec3& target) noexcept;

    // sets the target to position + forward
    void set_forward(const glm::vec3& forward) noexcept;

    // this method normalizes up for you
    void set_up(const glm::vec3& up) noexcept;

    // -----------------------------------------------------------------------------

    void set_perspective(const float field_of_view = 60.f,
                         const float aspect_ratio = 1.f,
                         const float near_plane = 0.1f,
                         const float far_plane = 1000.f) noexcept;

    void set_field_of_view(const float field_of_view) noexcept;

    // aspect_ratio = width / height
    void set_aspect_ratio(const float aspect_ratio) noexcept;

    void set_near_plane(const float near_plane) noexcept;

    void set_far_plane(const float far_plane) noexcept;

    float get_field_of_view() const noexcept;

    // High level operations
    // -----------------------------------------------------------------------------

    // Fitting the camera position and interest to see the bounding box
    // tight: Fit bounding box exactly, not tight: fit bounding sphere
    void
    look_at_bounding_box(const glm::vec3& box_min, const glm::vec3& box_max, bool tight = false);

    // Move your camera left-right (truck), up-down (pedestal) or in-out (dolly) according to
    // world-space coordinates, while the rotation stays the same. Note: dolly and truck requires a
    // certain distance to the object, else the looking direction cannot be calculated
    // Note that a positive dz moves back, because the camera looks to -z!
    void move(const float dx, const float dup, const float dz);

    // Move your camera left-right up-down or in-out (dolly) according to
    // camera coordinates, while the rotation stays the same.
    // Note that a positive dz moves back, because the camera looks to -z!
    void fly(const float dx, const float dy, const float dz);

    // Pan and tilt: rotate your camera horizontally (phi) or vertically (theta), while its base is
    // fixated on a
    // certain point. 2 * pi equals a full turn.
    void rotate(const float d_phi, const float d_theta);

    // Orbit around the "target" horizontally (phi) or vertically (theta).
    //  * pi equals a full turn.
    void orbit(const float d_phi, const float d_theta);

  private:
    // VIEW
    //-------------------------------------------------

    glm::vec3 position; // Position of the camera
    glm::vec3 target;   // Position where the camera is looking at
    glm::vec3 up;       // Normalized(!) up vector where the camera is oriented

    // Increase whenever position, target or up changes
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
