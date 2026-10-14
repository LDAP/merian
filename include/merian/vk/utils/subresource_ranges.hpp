#pragma once

#include <vulkan/vulkan.hpp>

namespace merian {

inline vk::ImageAspectFlags format_aspect_flags(const vk::Format format) {
    switch (format) {
    case vk::Format::eD16Unorm:
    case vk::Format::eX8D24UnormPack32:
    case vk::Format::eD32Sfloat:
        return vk::ImageAspectFlagBits::eDepth;
    case vk::Format::eS8Uint:
        return vk::ImageAspectFlagBits::eStencil;
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    default:
        return vk::ImageAspectFlagBits::eColor;
    }
}

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
