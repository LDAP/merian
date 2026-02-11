#pragma once

#include <cstdint>
#include <stdexcept>
#include <vulkan/vulkan_core.h>

namespace merian {

using SpirvVersion = uint32_t;

#define MERIAN_SPIRV_MAKE_VERSION(major, minor)                                                    \
    ((((uint32_t)(major)) << 16) | (((uint32_t)(minor)) << 8))

#define MERIAN_SPIRV_VERSION_MAJOR(version) (((version) >> 16) & 0xFF)
#define MERIAN_SPIRV_VERSION_MINOR(version) (((version) >> 8) & 0xFF)

#define MERIAN_SPIRV_VERSION_1_0 MERIAN_SPIRV_MAKE_VERSION(1, 0)
#define MERIAN_SPIRV_VERSION_1_1 MERIAN_SPIRV_MAKE_VERSION(1, 1)
#define MERIAN_SPIRV_VERSION_1_2 MERIAN_SPIRV_MAKE_VERSION(1, 2)
#define MERIAN_SPIRV_VERSION_1_3 MERIAN_SPIRV_MAKE_VERSION(1, 3)
#define MERIAN_SPIRV_VERSION_1_4 MERIAN_SPIRV_MAKE_VERSION(1, 4)
#define MERIAN_SPIRV_VERSION_1_5 MERIAN_SPIRV_MAKE_VERSION(1, 5)
#define MERIAN_SPIRV_VERSION_1_6 MERIAN_SPIRV_MAKE_VERSION(1, 6)
#define MERIAN_SPIRV_VERSION_LATEST MERIAN_SPIRV_VERSION_1_6

inline SpirvVersion spirv_target_for_vulkan_api_version(const uint32_t vulkan_api_version) {
    if (vulkan_api_version >= VK_API_VERSION_1_3) {
        return MERIAN_SPIRV_VERSION_1_6;
    }
    if (vulkan_api_version >= VK_API_VERSION_1_2) {
        return MERIAN_SPIRV_VERSION_1_5;
    }
    if (vulkan_api_version >= VK_API_VERSION_1_1) {
        return MERIAN_SPIRV_VERSION_1_3;
    }
    return MERIAN_SPIRV_VERSION_1_0;
}

inline uint32_t vulkan_api_version_for_spriv_version(const SpirvVersion spirv_version) {
    if (spirv_version > MERIAN_SPIRV_VERSION_1_6) {
        throw std::invalid_argument{"unknown spirv version"};
    }
    if (spirv_version > MERIAN_SPIRV_VERSION_1_5) {
        return VK_API_VERSION_1_3;
    }
    if (spirv_version > MERIAN_SPIRV_VERSION_1_3) {
        return VK_API_VERSION_1_2;
    }
    if (spirv_version > MERIAN_SPIRV_VERSION_1_0) {
        return VK_API_VERSION_1_1;
    }
    return VK_API_VERSION_1_0;
}

} // namespace merian
