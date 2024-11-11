#pragma once

#include "merian/vk/extension/extension.hpp"
#include <spdlog/spdlog.h>

namespace merian {

enum ExtensionVkFloatAtomicsFlags : uint8_t {
    EXTENSION_VK_FLOAT_ATOMICS_FLAGS_IMAGE32 = 0x0,
    EXTENSION_VK_FLOAT_ATOMICS_FLAGS_BUFFER32 = 0x2,
    EXTENSION_VK_FLOAT_ATOMICS_FLAGS_BUFFER64 = 0x4,
    EXTENSION_VK_FLOAT_ATOMICS_FLAGS_SHARED32 = 0x8,
    EXTENSION_VK_FLOAT_ATOMICS_FLAGS_SHARED64 = 0x10,

};

class ExtensionVkFloatAtomics : public Extension {
  public:
    ExtensionVkFloatAtomics(const ExtensionVkFloatAtomicsFlags flags = EXTENSION_VK_FLOAT_ATOMICS_FLAGS_IMAGE32)
        : Extension("ExtensionVkFloatAtomics"), flags(flags) {}
    ~ExtensionVkFloatAtomics() override = default;
    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice /*unused*/) const override {
        return {
            VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        };
    }

    void* pnext_get_features_2(void* const p_next) override {
        supported_atomic_features.setPNext(p_next);
        return &supported_atomic_features;
    }

    void* pnext_device_create_info(void* const p_next) override {
        if ((flags & EXTENSION_VK_FLOAT_ATOMICS_FLAGS_IMAGE32) != 0) {
            if (supported_atomic_features.shaderImageFloat32Atomics == VK_TRUE) {
                SPDLOG_DEBUG("shaderImageFloat32Atomics supported. Enabling feature");
                enable_atomic_features.shaderImageFloat32Atomics = VK_TRUE;
            } else {
                SPDLOG_WARN("shaderImageFloat32Atomics not supported");
            }

            if (supported_atomic_features.shaderImageFloat32AtomicAdd == VK_TRUE) {
                SPDLOG_DEBUG("shaderImageFloat32AtomicAdd supported. Enabling feature");
                enable_atomic_features.shaderImageFloat32AtomicAdd = VK_TRUE;
            } else {
                SPDLOG_WARN("shaderImageFloat32AtomicAdd not supported");
            }
        }
        if ((flags & EXTENSION_VK_FLOAT_ATOMICS_FLAGS_BUFFER32) != 0) {
            if (supported_atomic_features.shaderBufferFloat32Atomics == VK_TRUE) {
                SPDLOG_DEBUG("shaderBufferFloat32Atomics supported. Enabling feature");
                enable_atomic_features.shaderBufferFloat32Atomics = VK_TRUE;
            } else {
                SPDLOG_WARN("shaderBufferFloat32Atomics not supported");
            }
            if (supported_atomic_features.shaderBufferFloat32AtomicAdd == VK_TRUE) {
                SPDLOG_DEBUG("shaderBufferFloat32AtomicAdd supported. Enabling feature");
                enable_atomic_features.shaderBufferFloat32AtomicAdd = VK_TRUE;
            } else {
                SPDLOG_WARN("shaderBufferFloat32AtomicAdd not supported");
            }
        }
        if ((flags & EXTENSION_VK_FLOAT_ATOMICS_FLAGS_BUFFER64) != 0) {
            if (supported_atomic_features.shaderBufferFloat64Atomics == VK_TRUE) {
                SPDLOG_DEBUG("shaderBufferFloat64Atomics supported. Enabling feature");
                enable_atomic_features.shaderBufferFloat64Atomics = VK_TRUE;
            } else {
                SPDLOG_WARN("shaderBufferFloat64Atomics not supported");
            }
            if (supported_atomic_features.shaderBufferFloat64AtomicAdd == VK_TRUE) {
                SPDLOG_DEBUG("shaderBufferFloat64AtomicAdd supported. Enabling feature");
                enable_atomic_features.shaderBufferFloat64AtomicAdd = VK_TRUE;
            } else {
                SPDLOG_WARN("shaderBufferFloat64AtomicAdd not supported");
            }
        }
        if ((flags & EXTENSION_VK_FLOAT_ATOMICS_FLAGS_SHARED32) != 0) {
            if (supported_atomic_features.shaderSharedFloat32Atomics == VK_TRUE) {
                SPDLOG_DEBUG("shaderSharedFloat32Atomics supported. Enabling feature");
                enable_atomic_features.shaderSharedFloat32Atomics = VK_TRUE;
            } else {
                SPDLOG_WARN("shaderSharedFloat32AtomicAdd not supported");
            }

            if (supported_atomic_features.shaderSharedFloat32AtomicAdd == VK_TRUE) {
                SPDLOG_DEBUG("shaderSharedFloat32AtomicAdd supported. Enabling feature");
                enable_atomic_features.shaderSharedFloat32AtomicAdd = VK_TRUE;
            } else {
                SPDLOG_WARN("shaderSharedFloat32AtomicAdd not supported");
            }
        }

        if ((flags & EXTENSION_VK_FLOAT_ATOMICS_FLAGS_SHARED64) != 0) {
            if (supported_atomic_features.shaderSharedFloat64Atomics == VK_TRUE) {
                SPDLOG_DEBUG("shaderSharedFloat64Atomics supported. Enabling feature");
                enable_atomic_features.shaderSharedFloat64Atomics = VK_TRUE;
            } else {
                SPDLOG_WARN("shaderSharedFloat64Atomics not supported");
            }

            if (supported_atomic_features.shaderSharedFloat64AtomicAdd == VK_TRUE) {
                SPDLOG_DEBUG("shaderSharedFloat64AtomicAdd supported. Enabling feature");
                enable_atomic_features.shaderSharedFloat64AtomicAdd = VK_TRUE;
            } else {
                SPDLOG_WARN("shaderSharedFloat64AtomicAdd not supported");
            }
        }

        enable_atomic_features.pNext = p_next;
        return &enable_atomic_features;
    }

  private:
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT supported_atomic_features;
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT enable_atomic_features;
    const ExtensionVkFloatAtomicsFlags flags;
};

} // namespace merian
