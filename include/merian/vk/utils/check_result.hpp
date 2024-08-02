#pragma once


#include <vulkan/vulkan.hpp>
#include <fmt/format.h>

inline void check_result(vk::Result result, std::string msg_on_error) {
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error{fmt::format("{} ({})", msg_on_error, vk::to_string(result))};
    }
}

inline void check_result(VkResult result, std::string msg_on_error) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error{fmt::format("{} ({})", msg_on_error, vk::to_string(vk::Result(result)))};
    }
}
