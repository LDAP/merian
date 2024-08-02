#pragma once

#include "glm/ext/matrix_transform.hpp"
#include "glm/glm.hpp"
#include <numeric>
#include <vulkan/vulkan.hpp>

namespace merian {

inline vk::TransformMatrixKHR transform_identity() noexcept {
    vk::TransformMatrixKHR transform;
    transform.matrix[0][0] = transform.matrix[1][1] = transform.matrix[2][2] = 1.0f;
    return transform;
}

inline vk::Offset3D extent_to_offset(const vk::Extent3D& extent) noexcept {
    return {(int32_t)extent.width, (int32_t)extent.height, (int32_t)extent.depth};
}

inline vk::Extent3D offset_to_extent(const vk::Offset3D& offset) noexcept {
    return {(uint32_t)offset.x, (uint32_t)offset.y, (uint32_t)offset.z};
}

inline vk::Extent2D multiply(const vk::Extent2D& a, const float b) {
    return {static_cast<uint32_t>(std::round(a.width * b)),
            static_cast<uint32_t>(std::round(a.height * b))};
}

inline vk::Extent3D multiply(const vk::Extent3D& a, const float b) {
    return {static_cast<uint32_t>(std::round(a.width * b)),
            static_cast<uint32_t>(std::round(a.height * b)),
            static_cast<uint32_t>(std::round(a.depth * b))};
}

inline vk::Extent3D min(const vk::Extent3D& a, const vk::Extent3D& b) noexcept {
    return {std::min(a.width, b.width), std::min(a.height, b.height), std::min(a.depth, b.depth)};
}

inline vk::Extent3D max(const vk::Extent3D& a, const vk::Extent3D& b) noexcept {
    return {std::max(a.width, b.width), std::max(a.height, b.height), std::max(a.depth, b.depth)};
}

inline vk::Extent3D add(const vk::Extent3D& a, const vk::Extent3D& b) noexcept {
    return {a.width + b.width, a.height + b.height, a.depth + b.depth};
}

inline vk::Offset3D add(const vk::Offset3D& a, const vk::Offset3D& b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

// Returns offsets that center region onto extent.
inline std::pair<vk::Offset3D, vk::Offset3D> center(const vk::Extent3D& extent,
                                                    const vk::Extent3D& region) noexcept {
    assert(region.width <= extent.width);
    assert(region.height <= extent.height);
    assert(region.depth <= extent.depth);

    const int32_t half_width_diff = (extent.width - region.width) / 2;
    const int32_t half_height_diff = (extent.height - region.height) / 2;
    const int32_t half_depth_diff = (extent.depth - region.depth) / 2;

    const vk::Offset3D lower = {half_width_diff, half_height_diff, half_depth_diff};
    const vk::Offset3D upper = {(int32_t)extent.width - half_width_diff,
                                (int32_t)extent.height - half_height_diff,
                                (int32_t)extent.depth - half_depth_diff};
    return std::make_pair(lower, upper);
}

// Fits src into dst and returns the new dst offsets, assumes both images to have extent 1 in z
// direction.
inline std::pair<vk::Offset3D, vk::Offset3D> fit(const vk::Offset3D& src_lower,
                                                 const vk::Offset3D& src_upper,
                                                 const vk::Offset3D& dst_lower,
                                                 const vk::Offset3D& dst_upper) noexcept {
    const int32_t src_dx = src_upper.x - src_lower.x;
    const int32_t src_dy = src_upper.y - src_lower.y;

    const int32_t dst_dx = dst_upper.x - dst_lower.x;
    const int32_t dst_dy = dst_upper.y - dst_lower.y;

    assert(src_dx > 0);
    assert(src_dy > 0);
    assert(dst_dx > 0);
    assert(dst_dy > 0);

    const float scale = std::min(dst_dx / (float)src_dx, dst_dy / (float)src_dy);
    const auto [ctr_lower, ctr_upper] =
        center(vk::Extent3D(dst_dx, dst_dy, 1),
               vk::Extent3D(std::round(src_dx * scale), std::round(src_dy * scale), 1));

    return std::make_pair(add(dst_lower, ctr_lower), add(dst_lower, ctr_upper));
}

// Rotate "pos" around "origin". right-left (phi), up-down(theta).
// Keeps the up direction valid.
inline void rotate_around(glm::vec3& pos,
                          const glm::vec3& origin,
                          const glm::vec3& up,
                          const float d_phi,
                          const float d_theta) noexcept {
    const glm::vec3 origin_to_pos(pos - origin);
    const glm::vec3 normalized_origin_to_pos = glm::normalize(origin_to_pos);

    // left-right, around axis up
    const glm::mat4 rot_phi = glm::rotate(glm::identity<glm::mat4>(), -d_phi, up);

    // up-down, around axis x
    const glm::vec3 x = glm::normalize(glm::cross(up, normalized_origin_to_pos));
    const glm::mat4 rot_theta = glm::rotate(glm::identity<glm::mat4>(), -d_theta, x);

    glm::vec3 rotated = rot_theta * glm::vec4(origin_to_pos, 0);

    if (glm::dot(x, glm::cross(up, rotated)) <= 0) {
        // only rotate left-right
        rotated = glm::normalize(rot_phi * glm::vec4(origin_to_pos, 0));
    } else {
        // additionally rotate up-down
        rotated = glm::normalize(rot_phi * glm::vec4(rotated, 0));
    }

    pos = origin + rotated * glm::length(origin_to_pos);
}

// Calculates the lowest common multiple of two numbers
inline uint32_t lcm(uint32_t a, uint32_t b) noexcept {
    return (a * b) / std::gcd(a, b);
}

// Calculates the lowest common multiple of numbers
inline uint32_t lcm(std::vector<uint32_t> numbers) noexcept {
    if (numbers.empty())
        return 0;
    if (numbers.size() == 1) {
        return numbers[0];
    }

    uint32_t cur = lcm(numbers[0], numbers[1]);

    for (uint32_t i = 2; i < numbers.size(); i++) {
        cur = lcm(cur, numbers[i]);
    }

    return cur;
}

} // namespace merian
