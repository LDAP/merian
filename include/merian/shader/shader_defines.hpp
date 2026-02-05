#pragma once

namespace merian {

// Shader macro definition prefixes
// Note: Same prefixes used by both PhysicalDevice (all supported) and Device (only enabled)
inline constexpr const char* SHADER_DEFINE_PREFIX_INSTANCE_EXT = "MERIAN_INSTANCE_EXT_SUPPORTED_";
inline constexpr const char* SHADER_DEFINE_PREFIX_DEVICE_EXT = "MERIAN_DEVICE_EXT_SUPPORTED_";
inline constexpr const char* SHADER_DEFINE_PREFIX_SPIRV_EXT = "MERIAN_SPIRV_EXT_SUPPORTED_";
inline constexpr const char* SHADER_DEFINE_PREFIX_SPIRV_CAP = "MERIAN_SPIRV_CAP_SUPPORTED_";

} // namespace merian
