#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/utils/vector_matrix.hpp"
#include "merian/utils/aabb.hpp"

#include <memory>

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
    // fov is horizontal
    Camera(const float3& position = float3(0),
           const float3& target = float3(1, 0, 0),
           const float3& up = float3(0, 0, 1),
           const float field_of_view = 90.f,
           const float aspect_ratio = 1.f,
           const float near_plane = 0.1f,
           const float far_plane = 1000.f);

    // -----------------------------------------------------------------------------

    const float4x4& get_view_matrix() noexcept;

    const float4x4& get_projection_matrix() noexcept;

    float4x4 get_view_projection_matrix() noexcept;

    // -----------------------------------------------------------------------------

    // Convenience method that checks if the camera changed
    // and updates the supplied ID to the current ID.
    bool has_changed_update(uint64_t& check_id) const noexcept;

    uint64_t get_change_id() const noexcept;

    // -----------------------------------------------------------------------------

    void look_at(const float3& position, const float3& target, const float3& up) noexcept;

    void look_at(const float3& position,
                 const float3& target,
                 const float3& up,
                 const float field_of_view) noexcept;

    void set_position(const float3& position) noexcept;

    void set_target(const float3& center) noexcept;

    // this method normalizes up for you
    void set_up(const float3& up) noexcept;

    const float3& get_position() const noexcept;

    const float3& get_target() const noexcept;

    const float3& get_up() const noexcept;

    // -----------------------------------------------------------------------------

    // field_of_view is horizontal, in degrees.
    void set_perspective(const float field_of_view = 90.f,
                         const float aspect_ratio = 1.f,
                         const float near_plane = 0.1f,
                         const float far_plane = 1000.f) noexcept;

    void set_field_of_view(const float field_of_view) noexcept;

    // aspect_ratio = width / height
    void set_aspect_ratio(const float aspect_ratio) noexcept;

    void set_near_plane(const float near_plane) noexcept;

    void set_far_plane(const float far_plane) noexcept;

    float get_field_of_view() const noexcept;

    float get_aspect_ratio() const noexcept;

    float get_near_plane() const noexcept;

    float get_far_plane() const noexcept;

    const float3& get_forward() noexcept;
    const float3& get_right() noexcept;

    void write_to(ShaderCursor cursor);

    // High level operations
    // -----------------------------------------------------------------------------

    // Fitting the camera position and interest to see the bounding box
    // tight: Fit bounding box exactly, not tight: fit bounding sphere
    void look_at_bounding_box(const AABB& aabb);

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

    // Orbit around the "center" horizontally (phi) or vertically (theta).
    //  * pi equals a full turn.
    void orbit(const float d_phi, const float d_theta);

    void properties(Properties& props);

  private:
    // VIEW
    //-------------------------------------------------

    float3 position; // Position of the camera (also referred to as eye)
    float3 target;   // Position where the camera is looking at (also referred to as center)
    float3 up;       // Normalized(!) up vector where the camera is oriented

    // Increase whenever eye, center or up changes
    uint32_t view_change_id = 0;

    // Cache

    // Do not use diectly
    float4x4 view_cache;
    uint32_t view_change_id_cache = 0;

    // PROJECTION
    //-------------------------------------------------

    float field_of_view; // horizontal FOV, in degrees
    float aspect_ratio;  // aspect_ratio = width / height
    float near_plane;
    float far_plane;

    // Increase whenever fov, aspect_ratio, near_plane or far_plane changes
    uint32_t projection_change_id = 0;

    // Cache

    // Do not use diectly
    float4x4 projection_cache;
    uint32_t projection_change_id_cache = 0;

    // RAY BASIS (depends on both view and projection)

    float3 forward_cache;
    float3 right_cache;
    uint64_t ray_basis_change_id_cache = 0;

    void recompute_ray_basis();
};
using CameraHandle = std::shared_ptr<Camera>;

} // namespace merian
