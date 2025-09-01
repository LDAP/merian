#pragma once

#include "merian-nodes/connectors/connector_utils.hpp"
#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/entry_point.hpp"

#include <optional>

namespace merian_nodes {

class SVGF : public Node {
  private:
    struct VarianceEstimatePushConstant {
        float normal_reject_cos = 0.8;
        float depth_accept = 10; // larger reuses more
        float spatial_falloff = 3.0;
        float spatial_bias = 8.0;
    };

    struct FilterPushConstant {
        float param_z = 10; // parameter for depth      = 1   larger blurs more
        float param_n = .8; // parameter for normals    cos(alpha) for lower threshold
        float param_l = 8;  // parameter for brightness = 4   larger blurs more
        float z_bias_normals = -1.0;
        float z_bias_depth = -1.0;
    };

    struct TAAPushConstant {
        float blend_alpha = 0.0;
        float rejection_threshold = 1.0;
    };

  public:
    SVGF(const ContextHandle& context,
         const ResourceAllocatorHandle& allocator,
         const std::optional<vk::Format> output_format = std::nullopt);

    ~SVGF();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;
    const std::optional<vk::Format> output_format;

    // depends on available shared memory
    uint32_t variance_estimate_local_size;
    uint32_t filter_local_size;

    const uint32_t taa_local_size = 32;

    VkSampledImageInHandle con_prev_out = VkSampledImageIn::compute_read("prev_out", 1);
    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read("src");
    VkSampledImageInHandle con_history = VkSampledImageIn::compute_read("history");
    VkSampledImageInHandle con_albedo = VkSampledImageIn::compute_read("albedo");
    VkSampledImageInHandle con_mv = VkSampledImageIn::compute_read("mv", 0, true);
    GBufferInHandle con_gbuffer = merian_nodes::GBufferIn::compute_read("gbuffer");
    GBufferInHandle con_prev_gbuffer = merian_nodes::GBufferIn::compute_read("prev_gbuffer", 1);

    ManagedVkImageOutHandle con_out;

    EntryPointHandle variance_estimate_module;
    EntryPointHandle filter_module;
    EntryPointHandle taa_module;

    VarianceEstimatePushConstant variance_estimate_pc;
    FilterPushConstant filter_pc;
    TAAPushConstant taa_pc;

    vk::ImageCreateInfo irr_create_info;

    PipelineHandle variance_estimate;
    std::vector<PipelineHandle> filters;
    PipelineHandle taa;

    int svgf_iterations = 0;

    DescriptorSetLayoutHandle ping_pong_layout;
    DescriptorPoolHandle filter_pool;
    struct EAWRes {
        TextureHandle ping_pong;
        TextureHandle gbuf_ping_pong;
        // Set reads from this resources and writes to i ^ 1
        DescriptorSetHandle set;
    };
    std::array<EAWRes, 2> ping_pong_res; // Ping pong sets

    int filter_type = 2;

    int taa_debug = 0;
    int taa_filter_prev = 0;
    int taa_clamping = 0;
    int taa_mv_sampling = 0;
    bool enable_mv = true;

    bool kaleidoscope = false;
    bool kaleidoscope_use_shmem = true;
};

} // namespace merian_nodes
