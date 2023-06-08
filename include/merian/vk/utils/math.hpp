#pragma once

#include <vulkan/vulkan.hpp>

namespace merian {

inline vk::TransformMatrixKHR transform_identity() {
    vk::TransformMatrixKHR transform;
    transform.matrix[0][0] = transform.matrix[1][1] = transform.matrix[2][2] = 1.0f;
    return transform;
}

} // namespace merian
