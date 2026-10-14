#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/nodes/svgf/options.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"

#include "merian/vk/pipeline/pipeline.hpp"

namespace merian {

// One level of the edge-avoiding filter. À-trous: Dammertz et al. 2010, Edge-Avoiding À-Trous
// Wavelet Transform for fast Global Illumination Filtering; schedule: Dolp, Hanika and Dachsbacher,
// A Fast GPU Schedule For À-Trous Wavelet-Based Denoisers.
class SVGFFilter : public Node {
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

    uint32_t get_local_size() const;

    void set_options(const SVGFOptions& options);

    void set_level(const uint32_t level, const uint32_t level_count);

    void set_parameters(const SVGFFilterParameters& parameters);

  private:
    SVGFOptions options;
    uint32_t level = 0;
    uint32_t level_count = 1;
    SVGFFilterParameters parameters;

    uint32_t local_size = 0;

    const VkSampledImageInHandle con_src = VkSampledImageIn::create();
    const VkSampledImageInHandle con_gbuf_src = VkSampledImageIn::create();
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
