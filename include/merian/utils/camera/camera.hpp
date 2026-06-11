#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/utils/aabb.hpp"
#include "merian/utils/vector_matrix.hpp"

#include <memory>

namespace merian {

// Local coordinate system is x: right, y: up, looking into -z.
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
    // field_of_view is vertical, in radians
    Camera(const float3& position = float3(0),
           const float3& target = float3(1, 0, 0),
           const float3& up = float3(0, 0, 1),
           const float field_of_view = radians(60.f),
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

    // Compares the view and projection parameters, excluding jitter.
    bool operator==(const Camera& other) const noexcept;

    // -----------------------------------------------------------------------------

    void look_at(const float3& position, const float3& target, const float3& up) noexcept;

    // field_of_view is vertical, in radians
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

    // field_of_view is vertical, in radians
    void set_perspective(const float field_of_view = radians(60.f),
                         const float aspect_ratio = 1.f,
                         const float near_plane = 0.1f,
                         const float far_plane = 1000.f) noexcept;

    // in radians
    void set_field_of_view_vertical(const float field_of_view) noexcept;

    // in radians, converted to vertical using the current aspect ratio
    void set_field_of_view_horizontal(const float field_of_view) noexcept;

    // aspect_ratio = width / height
    void set_aspect_ratio(const float aspect_ratio) noexcept;

    void set_near_plane(const float near_plane) noexcept;

    void set_far_plane(const float far_plane) noexcept;

    // in radians
    float get_field_of_view_vertical() const noexcept;

    // in radians, derived from vertical fov and aspect ratio
    float get_field_of_view_horizontal() const noexcept;

    float get_aspect_ratio() const noexcept;

    float get_near_plane() const noexcept;

    float get_far_plane() const noexcept;

    const float3& get_forward() noexcept;
    const float3& get_right() noexcept;

    // Sub-pixel jitter sequence applied to the camera (for TAA / temporal accumulation).
    enum class JitterSequence : uint32_t {
        None = 0,
        Halton,         // Halton (2, 3) low-discrepancy sequence (the TAA default)
        R2,             // Martin Roberts' R2 low-discrepancy sequence
        BlackmanHarris, // radially symmetric Blackman-Harris distributed offset
    };

    // Sub-pixel jitter in render-pixel units; matches DLSS / FSR2 input convention
    // (X right, Y down, typical range [-0.5, 0.5]). Default 0.
    void set_jitter(const float2& jitter) noexcept;

    // Selects the sub-pixel jitter sequence; None resets the current jitter to 0.
    void set_jitter_sequence(JitterSequence sequence) noexcept;

    // Advances the jitter to the given frame within the selected sequence (no-op for None).
    void advance_jitter(uint32_t frame_index) noexcept;

    JitterSequence get_jitter_sequence() const noexcept;

    const float2& get_jitter() const noexcept;

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

    // Move the camera position toward (negative) or away from (positive) the target along the view
    // axis, leaving the target fixed (changes the eye-to-target distance).
    void dolly(const float distance);

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

    float field_of_view; // vertical FOV, in radians
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

    float2 jitter = {0.0f, 0.0f};
    JitterSequence jitter_sequence = JitterSequence::None;

    void recompute_ray_basis();
};
using CameraHandle = std::shared_ptr<Camera>;

} // namespace merian
