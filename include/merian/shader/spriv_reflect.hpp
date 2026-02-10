#pragma once

#include <cassert>

// Enable SpvCapabilityToString and friends in spirv.h (must be before include).
#ifndef SPV_ENABLE_UTILITY_CODE
#define SPV_ENABLE_UTILITY_CODE
#endif

#include <spirv_reflect.h>

#include "merian/vk/extension/extension.hpp"

#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <vector>

namespace merian {

class SpirvReflect {

  public:
    SpirvReflect(const uint32_t spv[], const std::size_t spv_size) : spv(spv), spv_size(spv_size) {
        SpvReflectResult result = spvReflectCreateShaderModule(spv_size, spv, &module);
        assert(result == SPV_REFLECT_RESULT_SUCCESS);
    }

    ~SpirvReflect() {
        spvReflectDestroyShaderModule(&module);
    }

    // ---------------------------------------------------------

    // Returns SPIR-V capability name strings (e.g. "RayTracingKHR").
    // The returned pointers are string literals (always valid).
    std::vector<const char*> get_capabilities() const {
        std::vector<const char*> result;
        result.reserve(module.capability_count);
        for (uint32_t i = 0; i < module.capability_count; i++) {
            result.emplace_back(SpvCapabilityToString(module.capabilities[i].value));
        }
        return result;
    }

    // Returns SPIR-V extension name strings (e.g. "SPV_KHR_ray_tracing").
    // The returned pointers point into the original SPV data passed to the constructor;
    std::vector<const char*> get_extensions() const {
        std::vector<const char*> result;

        // Skip the 5-word SPIR-V header.
        uint32_t offset = 5;
        while (offset < spv_size / 4) {
            const uint32_t instruction = spv[offset];
            const uint32_t op = instruction & 0xFFFF;
            const uint32_t len = instruction >> 16;

            if (len == 0)
                break;

            // OpExtension = 10
            if (op == SpvOpExtension && len >= 2) {
                result.emplace_back(reinterpret_cast<const char*>(&spv[offset + 1]));
            }

            // OpMemoryModel = 14 marks end of preamble.
            if (op == 14)
                break;

            offset += len;
        }

        return result;
    }

    // Checks if the SPIR-V module's capabilities and extensions are supported by the physical
    // device. Returns a DeviceSupportInfo with the supported flag and required capabilities/
    // extensions populated.
    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) const {
        DeviceSupportInfo info;

        const auto& supported_caps = query_info.physical_device->get_supported_spirv_capabilities();
        const auto& supported_exts = query_info.physical_device->get_supported_spirv_extensions();

        std::vector<const char*> missing_caps;
        for (const char* cap : get_capabilities()) {
            auto it = supported_caps.find(cap);
            if (it != supported_caps.end()) {
                info.required_spirv_capabilities.emplace_back(it->c_str());
            } else {
                missing_caps.emplace_back(cap);
            }
        }

        std::vector<const char*> missing_exts;
        for (const char* ext : get_extensions()) {
            auto it = supported_exts.find(ext);
            if (it != supported_exts.end()) {
                info.required_spirv_extensions.emplace_back(it->c_str());
            } else {
                missing_exts.emplace_back(ext);
            }
        }

        if (!missing_caps.empty() || !missing_exts.empty()) {
            info.supported = false;
            std::vector<std::string> parts;
            if (!missing_caps.empty())
                parts.emplace_back(
                    fmt::format("missing SPIR-V capabilities: {}", fmt::join(missing_caps, ", ")));
            if (!missing_exts.empty())
                parts.emplace_back(
                    fmt::format("missing SPIR-V extensions: {}", fmt::join(missing_exts, ", ")));
            info.unsupported_reason = fmt::format("{}", fmt::join(parts, "; "));
        }

        return info;
    }

  private:
    const uint32_t* spv;
    std::size_t spv_size;
    SpvReflectShaderModule module;
};

} // namespace merian
