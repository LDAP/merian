#pragma once

#include "merian/vk/context.hpp"

namespace merian {

inline void check_result(vk::Result result, const std::string& msg_on_error) {
    VulkanException::throw_if_no_success(result, msg_on_error);
}

inline void check_result(VkResult result, const std::string& msg_on_error) {
    VulkanException::throw_if_no_success(result, msg_on_error);
}

} // namespace merian
