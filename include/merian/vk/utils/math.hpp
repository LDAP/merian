#pragma once

#include "glm/ext/matrix_transform.hpp"
#include "glm/glm.hpp"
#include <vulkan/vulkan.hpp>

namespace merian {

inline vk::TransformMatrixKHR transform_identity() {
    vk::TransformMatrixKHR transform;
    transform.matrix[0][0] = transform.matrix[1][1] = transform.matrix[2][2] = 1.0f;
    return transform;
}

inline vk::Offset3D extent_to_offset(const vk::Extent3D& extent) {
    return {(int32_t)extent.width, (int32_t)extent.height, (int32_t)extent.depth};
}

inline vk::Extent3D offset_to_extent(const vk::Offset3D& offset) {
    return {(uint32_t)offset.x, (uint32_t)offset.y, (uint32_t)offset.z};
}

inline vk::Extent3D min(vk::Extent3D& a, vk::Extent3D& b) {
    return {std::min(a.width, b.width), std::min(a.height, b.height), std::min(a.depth, b.depth)};
}

// Returns the largest extent with aspect that is smaller than `extent`.
// Only width and height are considerd. For depth the original value is returned.
inline vk::Extent3D clamp_aspect(float aspect, const vk::Extent3D& extent) {
    const uint32_t width = std::min(extent.width, (uint32_t)(aspect * extent.height));
    const uint32_t height = std::min(extent.height, (uint32_t)(extent.width / aspect));
    return {width, height, extent.depth};
}

// Returns offsets that center region onto extent.
inline std::pair<vk::Offset3D, vk::Offset3D> center(vk::Extent3D extent, vk::Extent3D region) {
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

// Rotate "pos" around "origin". right-left (phi), up-down(theta).
// Keeps the up direction valid.
inline void
rotate_around(glm::vec3& pos, const glm::vec3& origin, const glm::vec3& up, const float d_phi, const float d_theta) {
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
        rotated =  glm::normalize(rot_phi * glm::vec4(origin_to_pos, 0));
    } else {
        // additionally rotate up-down
        rotated = glm::normalize(rot_phi * glm::vec4(rotated, 0));
    }
    
    pos = origin + rotated * glm::length(origin_to_pos);
}

} // namespace merian
