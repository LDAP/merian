#pragma once

#include "merian-nodes/connectors/vk_buffer_in.hpp"
#include "merian-nodes/connectors/vk_image_in.hpp"

#include "merian-nodes/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

#include <optional>

namespace merian_nodes {

class Accumulate : public Node {
  private:
    struct QuartilePushConstant {
        float firefly_percentile_lower = 0.25;
        float firefly_percentile_upper = 0.90;

        float adaptive_alpha_percentile_lower = 0.05;
        float adaptive_alpha_percentile_upper = 0.95;
    };

    struct FilterPushConstant {
        int32_t firefly_filter_enable = 0;
        float firefly_bias = 0.2;
        float firefly_ipr_factor = 50;

        float firefly_hard_clamp = INFINITY;

        float accum_alpha = 0.0;
        float accum_max_hist = INFINITY;
        float normal_reject_cos = 0.8;
        float depth_reject_percent = 0.02;
        int32_t clear = 0;

        float adaptive_alpha_reduction = 0.0;
        float adaptive_alpha_ipr_factor = 1.5;
    };

  public:
    Accumulate(const SharedContext context,
                   const ResourceAllocatorHandle allocator,
                   const std::optional<vk::Format> format = vk::Format::eR32G32B32A32Sfloat);

    ~Accumulate();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    NodeStatusFlags on_connected(const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

    NodeStatusFlags configuration([[maybe_unused]] Configuration& config) override;

    // Clears the accumulation buffer at the next iteration.
    void request_clear();

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;
    const std::optional<vk::Format> format;

    static constexpr uint32_t percentile_local_size_x = 8;
    static constexpr uint32_t percentile_local_size_y = 8;
    static constexpr uint32_t filter_local_size_x = 16;
    static constexpr uint32_t filter_local_size_y = 16;

    // Graph IO
    VkImageInHandle con_prev_accum = VkImageIn::compute_read("prev_accum", 1);
    VkImageInHandle con_prev_moments = VkImageIn::compute_read("prev_moments", 1);
    VkImageInHandle con_irr_in = VkImageIn::compute_read("irr");
    VkImageInHandle con_mv = VkImageIn::compute_read("mv");
    VkImageInHandle con_moments_in = VkImageIn::compute_read("moments_in");

    VkBufferInHandle con_gbuf = VkBufferIn::compute_read("gbuf");
    VkBufferInHandle con_prev_gbuf = VkBufferIn::compute_read("prev_gbuf", 1);

    VkImageOutHandle con_irr_out;
    VkImageOutHandle con_moments_out;

    vk::ImageCreateInfo irr_create_info;

    uint32_t percentile_group_count_x;
    uint32_t percentile_group_count_y;
    uint32_t filter_group_count_x;
    uint32_t filter_group_count_y;

    TextureHandle percentile_texture;

    ShaderModuleHandle percentile_module;
    ShaderModuleHandle accumulate_module;

    FilterPushConstant accumulate_pc;
    QuartilePushConstant percentile_pc;

    PipelineHandle calculate_percentiles;
    PipelineHandle accumulate;

    DescriptorSetLayoutHandle percentile_desc_layout;
    DescriptorPoolHandle percentile_desc_pool;
    DescriptorSetHandle percentile_set;
    DescriptorSetLayoutHandle accumulate_desc_layout;
    DescriptorPoolHandle accumulate_desc_pool;
    DescriptorSetHandle accumulate_set;

    bool clear = false;
    int filter_mode = 0;
    int extended_search = 1;
    int reuse_border = 0;
};

} // namespace merian_nodes
