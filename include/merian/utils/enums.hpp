#pragma once

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <fmt/format.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace merian {

template <typename VK_ENUM_TYPE> uint32_t enum_size();

template <typename VK_ENUM_TYPE> const VK_ENUM_TYPE* enum_values();

template <typename VK_ENUM_TYPE> std::string enum_to_string(VK_ENUM_TYPE value);

// The values of a sequential enum, in declaration order.
template <typename ENUM_TYPE> std::vector<ENUM_TYPE> enum_value_list() {
    std::vector<ENUM_TYPE> values;
    for (uint32_t i = 0; i < enum_size<ENUM_TYPE>(); i++) {
        values.emplace_back(static_cast<ENUM_TYPE>(i));
    }
    return values;
}

// Display strings for a MERIAN_ENUM value list.
inline std::vector<std::string> enum_names(const std::string_view values) {
    std::vector<std::string> names;
    for (std::size_t begin = 0; begin <= values.size();) {
        const std::size_t end = std::min(values.find(',', begin), values.size());
        std::string name;
        for (const char character : values.substr(begin, end - begin)) {
            if (character == '_') {
                name.push_back(' ');
            } else if (std::isspace(static_cast<unsigned char>(character)) == 0) {
                name.push_back(
                    static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
            }
        }
        names.emplace_back(std::move(name));
        begin = end + 1;
    }
    return names;
}

// Declares an enum together with the specializations config_enum needs. The display string of a
// value is its name, lowercased with underscores as spaces. Values are sequential, so do not
// assign any. Use it in namespace merian.
#define MERIAN_ENUM(TYPE, ...)                                                                     \
    enum class TYPE : uint32_t { __VA_ARGS__, MERIAN_ENUM_COUNT };                                 \
    template <> inline uint32_t enum_size<TYPE>() {                                                \
        return static_cast<uint32_t>(TYPE::MERIAN_ENUM_COUNT);                                     \
    }                                                                                              \
    template <> inline const TYPE* enum_values<TYPE>() {                                           \
        static const std::vector<TYPE> values = enum_value_list<TYPE>();                           \
        return values.data();                                                                      \
    }                                                                                              \
    template <> inline std::string enum_to_string<TYPE>(const TYPE value) {                        \
        static const std::vector<std::string> names = enum_names(#__VA_ARGS__);                    \
        assert(names.size() == enum_size<TYPE>());                                                 \
        return names.at(static_cast<std::size_t>(value));                                          \
    }

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
