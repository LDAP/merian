#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/nodes/svgf/options.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"

#include "merian/vk/pipeline/pipeline.hpp"

namespace merian {

// Estimates the variance of the accumulated irradiance and prepares the input of the first filter
// level.
class SVGFVarianceEstimate : public Node {
  private:
    struct PushConstant {
        float normal_reject_cos = 0.8;
        // larger reuses more
        float depth_accept = 10;
        float spatial_falloff = 3.0;
        float spatial_bias = 8.0;
    };

  public:
    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                 const NodeIO& io,
                                 const NodeConnectionInfo& info,
                                 Submission& submission) override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

    void set_options(const SVGFOptions& options);

    // Without filter levels the source is copied.
    void set_filter_levels(const uint32_t filter_levels);

    // Undefined keeps the format of the source.
    void set_format(const vk::Format format);

  private:
    SVGFOptions options;
    uint32_t filter_levels = 0;
    vk::Format format = vk::Format::eUndefined;
    PushConstant pc;

    uint32_t local_size = 0;
    vk::Extent3D extent;

    const VkSampledImageInHandle con_src = VkSampledImageIn::create();
    const VkSampledImageInHandle con_history = VkSampledImageIn::create();
    const GBufferInHandle con_gbuffer =
        GBufferIn::create({{{GBufferField::Normal, GBufferField::LinearZ, GBufferField::GradZ,
                             GBufferField::DeltaZ}}});

    ManagedVkImageOutHandle con_out;
    ManagedVkImageOutHandle con_gbuf;

    SlangProgramEntryPointHandle entry_point;
    PipelineHandle pipeline;
    ShaderObjectHandle globals;
};

} // namespace merian
