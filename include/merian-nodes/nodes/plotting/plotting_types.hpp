//
// Created by oschdi on 1/17/25.
//

#ifndef PLOTTING_TYPES_HPP
#define PLOTTING_TYPES_HPP

#include <array>

namespace merian {
enum PlottingType {
    INT_16,
    INT_32,
    FLOAT_32,
    FLOAT_16,
};

static constexpr std::array<PlottingType, 4> plottingType_values = {
    PlottingType::INT_16,
    PlottingType::INT_32,
    PlottingType::FLOAT_16,
    PlottingType::FLOAT_32
};

template<> inline uint32_t enum_size<PlottingType>() {
    return plottingType_values.size();
}

template<> inline const PlottingType* enum_values<PlottingType>() {
    return plottingType_values.data();
}

std::string inline to_string(PlottingType value) {
    switch (value) {
        case PlottingType::INT_16: return "Int 16";
        case PlottingType::INT_32: return "Int 32";
        case PlottingType::FLOAT_16: return "Float 16";
        case PlottingType::FLOAT_32: return "Float 32";
        default: return "invalid plotting type";
    }
}

template<> inline std::string enum_to_string<PlottingType>(PlottingType value) {
    return to_string(value);
}
}
#endif //PLOTTING_TYPES_HPP
