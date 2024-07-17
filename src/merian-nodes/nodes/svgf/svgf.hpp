#pragma once

#include "merian-nodes/connectors/managed_vk_buffer_in.hpp"
#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/graph/node.hpp"

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
    SVGF(const ContextHandle context,
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

    NodeStatusFlags properties(Properties& config) override;

  private:
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;
    const std::optional<vk::Format> output_format;

    // depends on available shared memory
    const uint32_t variance_estimate_local_size_x;
    const uint32_t variance_estimate_local_size_y;
    static constexpr uint32_t local_size_x = 32;
    static constexpr uint32_t local_size_y = 32;

    ManagedVkImageInHandle con_prev_out = ManagedVkImageIn::compute_read("prev_out", 1);
    ManagedVkImageInHandle con_irr = ManagedVkImageIn::compute_read("irr");
    ManagedVkImageInHandle con_moments = ManagedVkImageIn::compute_read("moments");
    ManagedVkImageInHandle con_albedo = ManagedVkImageIn::compute_read("albedo");
    ManagedVkImageInHandle con_mv = ManagedVkImageIn::compute_read("mv");
    ManagedVkBufferInHandle con_gbuffer = ManagedVkBufferIn::compute_read("gbuffer");
    ManagedVkBufferInHandle con_prev_gbuffer = ManagedVkBufferIn::compute_read("prev_gbuffer", 1);

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

    VkBool32 filter_variance = false;
    int filter_type = 0;

    int taa_debug = 0;
    int taa_filter_prev = false;
    int taa_clamping = 0;
    int taa_mv_sampling = 0;
};

} // namespace merian_nodes
