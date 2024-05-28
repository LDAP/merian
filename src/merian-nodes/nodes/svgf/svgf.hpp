#pragma once

#include "merian-nodes/graph/node.hpp"
#include "merian-nodes/connectors/vk_image_in.hpp"
#include "merian-nodes/connectors/vk_buffer_in.hpp"

#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

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
    SVGF(const SharedContext context,
             const ResourceAllocatorHandle allocator,
             const std::optional<vk::Format> output_format = std::nullopt);

    ~SVGF();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    NodeStatusFlags on_connected(const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

    NodeStatusFlags configuration(Configuration& config) override;

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;
    const std::optional<vk::Format> output_format;

    // depends on available shared memory
    const uint32_t variance_estimate_local_size_x;
    const uint32_t variance_estimate_local_size_y;
    static constexpr uint32_t local_size_x = 32;
    static constexpr uint32_t local_size_y = 32;

    VkImageInHandle con_prev_out = VkImageIn::compute_read("prev_out", 1);
    VkImageInHandle con_irr = VkImageIn::compute_read("irr");
    VkImageInHandle con_moments = VkImageIn::compute_read("moments");
    VkImageInHandle con_albedo = VkImageIn::compute_read("albedo");
    VkImageInHandle con_mv = VkImageIn::compute_read("mv");
    VkBufferInHandle con_gbuffer = VkBufferIn::compute_read("gbuffer");
    VkBufferInHandle con_prev_gbuffer = VkBufferIn::compute_read("prev_gbuffer", 1);

    ShaderModuleHandle variance_estimate_module;
    ShaderModuleHandle filter_module;
    ShaderModuleHandle taa_module;

    VarianceEstimatePushConstant variance_estimate_pc;
    FilterPushConstant filter_pc;
    TAAPushConstant taa_pc;

    vk::ImageCreateInfo irr_create_info;

    PipelineHandle variance_estimate;
    std::vector<PipelineHandle> filters;
    PipelineHandle taa;

    uint32_t group_count_x;
    uint32_t group_count_y;

    int svgf_iterations = 0;

    DescriptorSetLayoutHandle ping_pong_layout;
    DescriptorPoolHandle filter_pool;
    struct EAWRes {
        TextureHandle ping_pong;
        // Set reads from this resources and writes to i ^ 1
        DescriptorSetHandle set;
    };
    std::array<EAWRes, 2> ping_pong_res; // Ping pong sets

    int filter_variance = 0;
    int filter_type = 0;

    int taa_debug = 0;
    int taa_filter_prev = 0;
    int taa_clamping = 0;
    int taa_mv_sampling = 0;
};

} // namespace merian_nodes
