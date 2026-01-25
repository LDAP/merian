#pragma once

#include <cstdint>
#include <fmt/format.h>
#include <stdexcept>
#include <string>

namespace merian {

template <typename VK_ENUM_TYPE> uint32_t enum_size();

template <typename VK_ENUM_TYPE> const VK_ENUM_TYPE* enum_values();

template <typename VK_ENUM_TYPE> std::string enum_to_string(VK_ENUM_TYPE value);

template <typename VK_ENUM_TYPE> VK_ENUM_TYPE enum_from_string(const std::string value) {
    for (const VK_ENUM_TYPE* v = enum_values<VK_ENUM_TYPE>();
         v < enum_values<VK_ENUM_TYPE>() + enum_size<VK_ENUM_TYPE>(); v++) {
        if (enum_to_string(*v) == value) {
            return *v;
        }
    }

    throw std::invalid_argument{fmt::format("Value '{}' does not exists.", value)};
}

} // namespace merian
