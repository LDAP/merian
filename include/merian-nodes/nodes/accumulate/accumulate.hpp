#pragma once

#include "merian-nodes/connectors/managed_vk_buffer_in.hpp"
#include "merian-nodes/connectors/managed_vk_image_in.hpp"

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
        VkBool32 firefly_filter_enable = 0;
        float firefly_bias = 0.2;
        float firefly_ipr_factor = 50;

        float firefly_hard_clamp = INFINITY;

        float accum_alpha = 0.0;
        float accum_max_hist = INFINITY;
        float normal_reject_cos = 0.8;
        float depth_reject_percent = 0.02;
        VkBool32 clear = 0;

        float adaptive_alpha_reduction = 0.0;
        float adaptive_alpha_ipr_factor = 1.5;

        uint32_t iteration = 0;
    };

  public:
    Accumulate(const ContextHandle& context,
               const ResourceAllocatorHandle& allocator,
               const std::optional<vk::Format> format = vk::Format::eR32G32B32A32Sfloat);

    ~Accumulate();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void process(GraphRun& run,
                 const CommandBufferHandle& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

    NodeStatusFlags properties([[maybe_unused]] Properties& config) override;

    // Clears the accumulation buffer at the next iteration.
    void request_clear();

  private:
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;
    const std::optional<vk::Format> format;

    static constexpr uint32_t PERCENTILE_LOCAL_SIZE_X = 8;
    static constexpr uint32_t PERCENTILE_LOCAL_SIZE_Y = 8;
    static constexpr uint32_t FILTER_LOCAL_SIZE_X = 16;
    static constexpr uint32_t FILTER_LOCAL_SIZE_Y = 16;

    // Graph IO
    ManagedVkImageInHandle con_prev_accum = ManagedVkImageIn::compute_read("prev_accum", 1);
    ManagedVkImageInHandle con_prev_moments = ManagedVkImageIn::compute_read("prev_moments", 1);
    ManagedVkImageInHandle con_irr_in = ManagedVkImageIn::compute_read("irr");
    ManagedVkImageInHandle con_mv = ManagedVkImageIn::compute_read("mv", 0, true);
    ManagedVkImageInHandle con_moments_in = ManagedVkImageIn::compute_read("moments_in");

    ManagedVkBufferInHandle con_gbuf = ManagedVkBufferIn::compute_read("gbuf");
    ManagedVkBufferInHandle con_prev_gbuf = ManagedVkBufferIn::compute_read("prev_gbuf", 1);

    ManagedVkImageOutHandle con_irr_out;
    ManagedVkImageOutHandle con_moments_out;

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
    VkBool32 extended_search = VK_TRUE;
    VkBool32 reuse_border = VK_FALSE;
    bool enable_mv = VK_TRUE;

    std::string clear_event_listener_pattern = "/user/clear";
};

} // namespace merian_nodes
