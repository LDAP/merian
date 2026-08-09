// Parsing logic derived from pbrt-v4 (https://github.com/mmp/pbrt-v4),
// Copyright (c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys. Apache-2.0.
#pragma once

#include "merian/utils/vector_matrix.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace merian::pbrt {

struct ParsedParameter {
    std::string type;
    std::string name;
    std::vector<float> floats;
    std::vector<int32_t> ints;
    std::vector<std::string> strings;
    std::vector<uint8_t> bools;
};

class ParamDict {
  public:
    const ParsedParameter* find(const std::string_view name) const {
        for (const ParsedParameter& param : params) {
            if (param.name == name) {
                return &param;
            }
        }
        return nullptr;
    }

    float get_float(const std::string_view name, const float fallback) const {
        const ParsedParameter* param = find(name);
        return param != nullptr && !param->floats.empty() ? param->floats[0] : fallback;
    }

    int32_t get_int(const std::string_view name, const int32_t fallback) const {
        const ParsedParameter* param = find(name);
        return param != nullptr && !param->ints.empty() ? param->ints[0] : fallback;
    }

    bool get_bool(const std::string_view name, const bool fallback) const {
        const ParsedParameter* param = find(name);
        return param != nullptr && !param->bools.empty() ? param->bools[0] != 0 : fallback;
    }

    std::string get_string(const std::string_view name, const std::string& fallback) const {
        const ParsedParameter* param = find(name);
        return param != nullptr && !param->strings.empty() ? param->strings[0] : fallback;
    }

    std::vector<float3> get_vec3_list(const std::string_view name) const {
        const ParsedParameter* param = find(name);
        std::vector<float3> result;
        if (param != nullptr) {
            result.reserve(param->floats.size() / 3);
            for (size_t i = 0; i + 2 < param->floats.size(); i += 3) {
                result.emplace_back(param->floats[i], param->floats[i + 1], param->floats[i + 2]);
            }
        }
        return result;
    }

    std::vector<float2> get_vec2_list(const std::string_view name) const {
        const ParsedParameter* param = find(name);
        std::vector<float2> result;
        if (param != nullptr) {
            result.reserve(param->floats.size() / 2);
            for (size_t i = 0; i + 1 < param->floats.size(); i += 2) {
                result.emplace_back(param->floats[i], param->floats[i + 1]);
            }
        }
        return result;
    }

    const std::vector<int32_t>* get_int_list(const std::string_view name) const {
        const ParsedParameter* param = find(name);
        return param != nullptr ? &param->ints : nullptr;
    }

    std::vector<ParsedParameter> params;
};

} // namespace merian::pbrt
