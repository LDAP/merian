#pragma once

#include "merian-nodes/connectors/connector_utils.hpp"
#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"

#include "merian-nodes/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/shader/entry_point.hpp"
#include <optional>

namespace merian {

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
    Accumulate();

    ~Accumulate();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties([[maybe_unused]] Properties& config) override;

    // Clears the accumulation buffer at the next iteration.
    void request_clear();

  private:
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    std::optional<vk::Format> format = vk::Format::eR32G32B32A32Sfloat;

    static constexpr uint32_t PERCENTILE_LOCAL_SIZE_X = 8;
    static constexpr uint32_t PERCENTILE_LOCAL_SIZE_Y = 8;
    static constexpr uint32_t FILTER_LOCAL_SIZE_X = 16;
    static constexpr uint32_t FILTER_LOCAL_SIZE_Y = 16;

    // Graph IO
    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read("src");
    GBufferInHandle con_gbuf = GBufferIn::compute_read("gbuffer");
    VkSampledImageInHandle con_mv = VkSampledImageIn::compute_read("mv", 0, true);

    VkSampledImageInHandle con_prev_out = VkSampledImageIn::compute_read("prev_out", 1);
    GBufferInHandle con_prev_gbuf = GBufferIn::compute_read("prev_gbuffer", 1);
    VkSampledImageInHandle con_prev_history = VkSampledImageIn::compute_read("prev_history", 1);

    ManagedVkImageOutHandle con_out;
    ManagedVkImageOutHandle con_history;

    vk::ImageCreateInfo irr_create_info;

    uint32_t percentile_group_count_x;
    uint32_t percentile_group_count_y;
    uint32_t filter_group_count_x;
    uint32_t filter_group_count_y;

    TextureHandle percentile_texture;

    EntryPointHandle percentile_module;
    EntryPointHandle accumulate_module;

    FilterPushConstant accumulate_pc;
    QuartilePushConstant percentile_pc;

    PipelineHandle calculate_percentiles;
    PipelineHandle accumulate;

    DescriptorSetLayoutHandle percentile_desc_layout;
    DescriptorSetHandle percentile_set;
    DescriptorSetLayoutHandle accumulate_desc_layout;
    DescriptorSetHandle accumulate_set;

    bool clear = false;
    int filter_mode = 0;
    VkBool32 extended_search = VK_TRUE;
    VkBool32 reuse_border = VK_FALSE;
    bool enable_mv = VK_TRUE;

    std::string clear_event_listener_pattern = "/user/clear";
};

} // namespace merian
