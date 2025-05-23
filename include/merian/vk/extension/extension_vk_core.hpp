#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * Extension to configure core features.
 *
 */
class ExtensionVkCore : public Extension {
  public:
    class CoreFeatureContainer {
        friend ExtensionVkCore;

      public:
        operator const vk::PhysicalDeviceFeatures2&() const {
            return physical_device_features;
        }

        operator vk::PhysicalDeviceFeatures2&() {
            return physical_device_features;
        }

        operator const vk::PhysicalDeviceFeatures&() const {
            return physical_device_features.features;
        }

        operator vk::PhysicalDeviceFeatures&() {
            return physical_device_features.features;
        }

        const vk::PhysicalDeviceFeatures2& get_physical_device_features() const {
            return physical_device_features;
        }
        const vk::PhysicalDeviceVulkan11Features& get_physical_device_features_v11() const {
            return physical_device_features_v11;
        }
        const vk::PhysicalDeviceVulkan12Features& get_physical_device_features_v12() const {
            return physical_device_features_v12;
        }
        const vk::PhysicalDeviceVulkan13Features& get_physical_device_features_v13() const {
            return physical_device_features_v13;
        }

      private:
        vk::PhysicalDeviceFeatures2 physical_device_features;
        vk::PhysicalDeviceVulkan11Features physical_device_features_v11;
        vk::PhysicalDeviceVulkan12Features physical_device_features_v12;
        vk::PhysicalDeviceVulkan13Features physical_device_features_v13;
    };

  public:
    /* Configure using the pattern vkXX/featureName, where XX is 10,11,12,13.
     */
    ExtensionVkCore(const std::set<std::string>& required_features = {},
                    const std::set<std::string>& optional_features =
                        {

                            // VK 1.0

                            "vk10/robustBufferAccess",
                            // "vk10/fullDrawIndexUint32",
                            // "vk10/imageCubeArray",
                            // "vk10/independentBlend",
                            "vk10/geometryShader", "vk10/tessellationShader",
                            // "vk10/sampleRateShading",
                            // "vk10/dualSrcBlend",
                            // "vk10/logicOp",
                            // "vk10/multiDrawIndirect",
                            // "vk10/drawIndirectFirstInstance",
                            "vk10/depthClamp", "vk10/depthBiasClamp",
                            // "vk10/fillModeNonSolid",
                            // "vk10/depthBounds",
                            // "vk10/wideLines",
                            // "vk10/largePoints",
                            "vk10/alphaToOne",
                            // "vk10/multiViewport",
                            "vk10/samplerAnisotropy",
                            // "vk10/textureCompressionETC2",
                            // "vk10/textureCompressionASTC_LDR",
                            // "vk10/textureCompressionBC",
                            // "vk10/occlusionQueryPrecise",
                            // "vk10/pipelineStatisticsQuery",
                            "vk10/vertexPipelineStoresAndAtomics", "vk10/fragmentStoresAndAtomics",
                            // "vk10/shaderTessellationAndGeometryPointSize",
                            // "vk10/shaderImageGatherExtended",
                            // "vk10/shaderStorageImageExtendedFormats",
                            // "vk10/shaderStorageImageMultisample",
                            // "vk10/shaderStorageImageReadWithoutFormat",
                            // "vk10/shaderStorageImageWriteWithoutFormat",
                            "vk10/shaderUniformBufferArrayDynamicIndexing",
                            "vk10/shaderSampledImageArrayDynamicIndexing",
                            "vk10/shaderStorageBufferArrayDynamicIndexing",
                            "vk10/shaderStorageImageArrayDynamicIndexing",
                            "vk10/shaderClipDistance", "vk10/shaderCullDistance",
                            "vk10/shaderFloat64", "vk10/shaderInt64", "vk10/shaderInt16",
                            //"vk10/shaderResourceResidency",
                            "vk10/shaderResourceMinLod", "vk10/sparseBinding",
                            //"vk10/sparseResidencyBuffer",
                            // "vk10/sparseResidencyImage2D",
                            // "vk10/sparseResidencyImage3D",
                            // "vk10/sparseResidency2Samples",
                            // "vk10/sparseResidency4Samples",
                            // "vk10/sparseResidency8Samples",
                            // "vk10/sparseResidency16Samples",
                            // "vk10/sparseResidencyAliased",
                            // "vk10/variableMultisampleRate",
                            // "vk10/inheritedQueries",

                            // VK 1.1

                            "vk11/storageBuffer16BitAccess",

                            // VK 1.2

                            "vk12/scalarBlockLayout", "vk12/shaderFloat16",
                            "vk12/uniformAndStorageBuffer8BitAccess", "vk12/bufferDeviceAddress",
                            "vk12/runtimeDescriptorArray", "vk12/descriptorIndexing",
                            "vk12/shaderSampledImageArrayNonUniformIndexing",
                            "vk12/shaderStorageImageArrayNonUniformIndexing",
                            "vk12/shaderStorageBufferArrayNonUniformIndexing",
                            "vk12/shaderUniformBufferArrayNonUniformIndexing", "vk12/shaderInt8",
                            "vk12/timelineSemaphore", "vk12/hostQueryReset",

                            // VK 1.3

                            "vk13/robustImageAccess", "vk13/synchronization2", "vk13/maintenance4",
                            "vk13/subgroupSizeControl"

                        },
                    const std::vector<const char*>& device_extensions = {},
                    const std::vector<const char*>& instance_extensions = {},
                    const std::vector<const char*>& instance_layers = {});

    ~ExtensionVkCore();

    // --------------------------------------------------------------------------

    /* Extensions that should be enabled instance-wide. */
    std::vector<const char*> required_instance_extension_names() const override;
    /* Layers that should be enabled instance-wide. */
    std::vector<const char*> required_instance_layer_names() const override;
    /* Extensions that should be enabled device-wide. Note that on_physical_device_selected is
     * called before. */
    std::vector<const char*>
    required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const override;

    // --------------------------------------------------------------------------

    void* pnext_get_features_2(void* const p_next) override;

    bool extension_supported(const vk::Instance& /*unused*/,
                             const PhysicalDevice& physical_device,
                             const ExtensionContainer& /*unused*/,
                             const QueueInfo& /*unused*/) override;

    void* pnext_device_create_info(void* const p_next) override;

    void on_unsupported([[maybe_unused]] const std::string& reason) override;

    // --------------------------------------------------------------------------

    const CoreFeatureContainer& get_supported_features() const;

    const CoreFeatureContainer& get_enabled_features() const;

    void request_required_feature(const std::string& feature);

    void request_optional_feature(const std::string& feature);

  private:
    std::set<std::string> required_features;
    std::set<std::string> optional_features;
    const std::vector<const char*> device_extensions;
    const std::vector<const char*> instance_extensions;
    const std::vector<const char*> instance_layers;

    CoreFeatureContainer supported;
    CoreFeatureContainer enabled;
};

} // namespace merian
