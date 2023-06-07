#pragma once

#include <vulkan/vulkan.hpp>

namespace merian {

inline vk::ImageSubresourceRange
all_levels_and_layers(vk::ImageAspectFlags aspect_flags = vk::ImageAspectFlagBits::eColor) {
    return {aspect_flags, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
}

inline vk::ImageSubresourceRange
first_level_and_layer(vk::ImageAspectFlags aspect_flags = vk::ImageAspectFlagBits::eColor) {
    return {aspect_flags, 0, 1, 0, 1};
}

inline vk::ImageSubresourceLayers
all_layers(vk::ImageAspectFlags aspect_flags = vk::ImageAspectFlagBits::eColor) {
    return {aspect_flags, 0, 0, VK_REMAINING_ARRAY_LAYERS};
}

inline vk::ImageSubresourceLayers
first_layer(vk::ImageAspectFlags aspect_flags = vk::ImageAspectFlagBits::eColor) {
    return {aspect_flags, 0, 0, 1};
}

} // namespace merian
