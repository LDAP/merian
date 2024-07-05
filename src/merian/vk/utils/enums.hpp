#pragma once

#include <cstdint>
#include <string>

namespace merian {

template <typename VK_ENUM_TYPE> uint32_t enum_size();

template <typename VK_ENUM_TYPE> const VK_ENUM_TYPE* enum_values();

template <typename VK_ENUM_TYPE> std::string enum_to_string(VK_ENUM_TYPE value);

} // namespace merian
