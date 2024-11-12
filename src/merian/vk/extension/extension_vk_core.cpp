#include "merian/vk/extension/extension_vk_core.hpp"

namespace merian {

ExtensionVkCore::ExtensionVkCore(const std::set<std::string>& required_features,
                                 const std::set<std::string>& optional_features,
                                 const std::vector<const char*>& device_extensions,
                                 const std::vector<const char*>& instance_extensions,
                                 const std::vector<const char*>& instance_layers)
    : Extension("ExtensionVkCore"), required_features(required_features),
      optional_features(optional_features), device_extensions(device_extensions),
      instance_extensions(instance_extensions), instance_layers(instance_layers) {}

ExtensionVkCore::~ExtensionVkCore() {}

// --------------------------------------------------------------------------

/* Extensions that should be enabled instance-wide. */
std::vector<const char*> ExtensionVkCore::required_instance_extension_names() const {
    return instance_extensions;
}
/* Layers that should be enabled instance-wide. */
std::vector<const char*> ExtensionVkCore::required_instance_layer_names() const {
    return instance_layers;
}
/* Extensions that should be enabled device-wide. Note that on_physical_device_selected is
 * called before. */
std::vector<const char*>
ExtensionVkCore::required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const {
    return device_extensions;
}

// --------------------------------------------------------------------------

void* ExtensionVkCore::pnext_get_features_2(void* const p_next) {
    // ^
    supported.physical_device_features_v13.setPNext(p_next);
    // ^
    supported.physical_device_features_v12.setPNext(&supported.physical_device_features_v13);
    // ^
    supported.physical_device_features_v11.setPNext(&supported.physical_device_features_v12);
    // ^
    return &supported.physical_device_features_v11;
}

bool ExtensionVkCore::extension_supported(const PhysicalDevice& physical_device,
                                          const ExtensionContainer& /*unused*/) {
    bool all_required_supported = true;

    supported.physical_device_features = physical_device.physical_device_features;

#define MERIAN_EXT_ENABLE_IF_REQUESTED(prefix, name, supported, enabled)                           \
    do {                                                                                           \
        const bool required = required_features.contains(#prefix "/" #name);                       \
        const bool optional = optional_features.contains(#prefix "/" #name);                       \
        if ((supported).name && (required || optional)) {                                          \
            (enabled).name = VK_TRUE;                                                              \
            SPDLOG_DEBUG("enable core feature {}/{}", #prefix, #name);                             \
        }                                                                                          \
        if (required && !(supported).name) {                                                       \
            all_required_supported = false;                                                        \
            SPDLOG_ERROR("core feature {}/{} required but not supported", #prefix, #name);         \
        }                                                                                          \
        if (optional && !(supported).name) {                                                       \
            SPDLOG_DEBUG("core feature {}/{} optionally requested but not supported", #prefix,     \
                         #name);                                                                   \
        }                                                                                          \
        required_features.erase(#prefix "/" #name);                                                \
        optional_features.erase(#prefix "/" #name);                                                \
    } while (0)

    // VK 1.0

    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, robustBufferAccess,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, fullDrawIndexUint32,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, imageCubeArray,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, independentBlend,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, geometryShader,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, tessellationShader,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, sampleRateShading,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, dualSrcBlend, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, logicOp, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, multiDrawIndirect,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, drawIndirectFirstInstance,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, depthClamp, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, depthBiasClamp,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, fillModeNonSolid,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, depthBounds, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, wideLines, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, largePoints, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, alphaToOne, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, multiViewport, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, samplerAnisotropy,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, textureCompressionETC2,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, textureCompressionASTC_LDR,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, textureCompressionBC,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, occlusionQueryPrecise,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, pipelineStatisticsQuery,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, vertexPipelineStoresAndAtomics,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, fragmentStoresAndAtomics,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderTessellationAndGeometryPointSize,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderImageGatherExtended,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderStorageImageExtendedFormats,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderStorageImageMultisample,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderStorageImageReadWithoutFormat,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderStorageImageWriteWithoutFormat,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderUniformBufferArrayDynamicIndexing,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderSampledImageArrayDynamicIndexing,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderStorageBufferArrayDynamicIndexing,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderStorageImageArrayDynamicIndexing,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderClipDistance,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderCullDistance,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderFloat64, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderInt64, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderInt16, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderResourceResidency,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, shaderResourceMinLod,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, sparseBinding, supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, sparseResidencyBuffer,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, sparseResidencyImage2D,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, sparseResidencyImage3D,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, sparseResidency2Samples,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, sparseResidency4Samples,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, sparseResidency8Samples,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, sparseResidency16Samples,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, sparseResidencyAliased,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, variableMultisampleRate,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk10, inheritedQueries,
                                   supported.physical_device_features.features,
                                   enabled.physical_device_features.features);

    // VK 1.1

    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, storageBuffer16BitAccess,
                                   supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, uniformAndStorageBuffer16BitAccess,
                                   supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, storagePushConstant16,
                                   supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, storageInputOutput16,
                                   supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, multiview, supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, multiviewGeometryShader,
                                   supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, multiviewTessellationShader,
                                   supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, variablePointersStorageBuffer,
                                   supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, variablePointers, supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, protectedMemory, supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, samplerYcbcrConversion,
                                   supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk11, shaderDrawParameters,
                                   supported.physical_device_features_v11,
                                   enabled.physical_device_features_v11);

    // VK 1.2

    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, samplerMirrorClampToEdge,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, drawIndirectCount, supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, storageBuffer8BitAccess,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, uniformAndStorageBuffer8BitAccess,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, storagePushConstant8,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderBufferInt64Atomics,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderSharedInt64Atomics,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderFloat16, supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderInt8, supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, descriptorIndexing, supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderInputAttachmentArrayDynamicIndexing,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderUniformTexelBufferArrayDynamicIndexing,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderStorageTexelBufferArrayDynamicIndexing,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderUniformBufferArrayNonUniformIndexing,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderSampledImageArrayNonUniformIndexing,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderStorageBufferArrayNonUniformIndexing,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderStorageImageArrayNonUniformIndexing,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderInputAttachmentArrayNonUniformIndexing,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderUniformTexelBufferArrayNonUniformIndexing,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderStorageTexelBufferArrayNonUniformIndexing,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, descriptorBindingUniformBufferUpdateAfterBind,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, descriptorBindingSampledImageUpdateAfterBind,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, descriptorBindingStorageImageUpdateAfterBind,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, descriptorBindingStorageBufferUpdateAfterBind,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, descriptorBindingUniformTexelBufferUpdateAfterBind,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, descriptorBindingStorageTexelBufferUpdateAfterBind,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, descriptorBindingUpdateUnusedWhilePending,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, descriptorBindingPartiallyBound,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, descriptorBindingVariableDescriptorCount,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, runtimeDescriptorArray,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, samplerFilterMinmax,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, scalarBlockLayout, supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, imagelessFramebuffer,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, uniformBufferStandardLayout,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderSubgroupExtendedTypes,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, separateDepthStencilLayouts,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, hostQueryReset, supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, timelineSemaphore, supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, bufferDeviceAddress,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, bufferDeviceAddressCaptureReplay,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, bufferDeviceAddressMultiDevice,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, vulkanMemoryModel, supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, vulkanMemoryModelDeviceScope,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, vulkanMemoryModelAvailabilityVisibilityChains,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderOutputViewportIndex,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, shaderOutputLayer, supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk12, subgroupBroadcastDynamicId,
                                   supported.physical_device_features_v12,
                                   enabled.physical_device_features_v12);

    // VK 1.3

    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, robustImageAccess, supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, inlineUniformBlock, supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, descriptorBindingInlineUniformBlockUpdateAfterBind,
                                   supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, pipelineCreationCacheControl,
                                   supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, privateData, supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, shaderDemoteToHelperInvocation,
                                   supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, shaderTerminateInvocation,
                                   supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, subgroupSizeControl,
                                   supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, computeFullSubgroups,
                                   supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, synchronization2, supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, textureCompressionASTC_HDR,
                                   supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, shaderZeroInitializeWorkgroupMemory,
                                   supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, dynamicRendering, supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, shaderIntegerDotProduct,
                                   supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);
    MERIAN_EXT_ENABLE_IF_REQUESTED(vk13, maintenance4, supported.physical_device_features_v13,
                                   enabled.physical_device_features_v13);

    for (const auto& unknown_feature : optional_features) {
        SPDLOG_WARN("feature {} is unknown", unknown_feature);
    }
    for (const auto& unknown_feature : required_features) {
        SPDLOG_WARN("feature {} is unknown", unknown_feature);
    }

#undef MERIAN_EXT_ENABLE_IF_REQUESTED
    return all_required_supported;
}

void* ExtensionVkCore::pnext_device_create_info(void* const p_next) {
    enabled.physical_device_features_v13.setPNext(p_next);
    // ^
    enabled.physical_device_features_v12.setPNext(&enabled.physical_device_features_v13);
    // ^
    enabled.physical_device_features_v11.setPNext(&enabled.physical_device_features_v12);
    // ^
    enabled.physical_device_features.setPNext(&enabled.physical_device_features_v11);
    // ^
    return &enabled.physical_device_features;
}

void ExtensionVkCore::on_unsupported([[maybe_unused]] const std::string& reason) {
    throw std::runtime_error{"ExtensionVkCore configuration not supported!"};
}

// --------------------------------------------------------------------------

const ExtensionVkCore::CoreFeatureContainer& ExtensionVkCore::get_supported_features() const {
    return supported;
}

const ExtensionVkCore::CoreFeatureContainer& ExtensionVkCore::get_enabled_features() const {
    return enabled;
}

void ExtensionVkCore::request_required_feature(const std::string& feature) {
    required_features.insert(feature);
}

void ExtensionVkCore::request_optional_feature(const std::string& feature) {
    optional_features.insert(feature);
}

} // namespace merian
