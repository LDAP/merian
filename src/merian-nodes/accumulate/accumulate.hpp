#pragma once

#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class AccumulateNode : public Node {
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
    AccumulateNode(const SharedContext context,
                   const ResourceAllocatorHandle allocator,
                   const std::optional<vk::Format> format = vk::Format::eR32G32B32A32Sfloat);

    ~AccumulateNode();

    std::string name() override {
        return "Accumulate";
    };

    std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(
        const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
        const std::vector<NodeOutputDescriptorBuffer>& connected_buffer_outputs) override;

    void cmd_build(const vk::CommandBuffer& cmd,
                   const std::vector<std::vector<ImageHandle>>& image_inputs,
                   const std::vector<std::vector<BufferHandle>>& buffer_inputs,
                   const std::vector<std::vector<ImageHandle>>& image_outputs,
                   const std::vector<std::vector<BufferHandle>>& buffer_outputs) override;

    void cmd_process(const vk::CommandBuffer& cmd,
                     GraphRun& run,
                     const uint32_t set_index,
                     const std::vector<ImageHandle>& image_inputs,
                     const std::vector<BufferHandle>& buffer_inputs,
                     const std::vector<ImageHandle>& image_outputs,
                     const std::vector<BufferHandle>& buffer_outputs) override;

    void get_configuration(Configuration& config, bool& needs_rebuild) override;

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

    std::vector<TextureHandle> graph_textures;
    std::vector<DescriptorSetHandle> graph_sets;
    DescriptorSetLayoutHandle graph_layout;
    DescriptorPoolHandle graph_pool;

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

} // namespace merian
