#pragma once

#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class FireflyFilterNode : public Node {
  private:
    struct QuartilePushConstant {
      float percentile_lower = 0.25;
      float percentile_upper = 0.90;
    };

    struct FilterPushConstant {
        int32_t enabled = 1;
        float bias = 0.2;
        float ipr_factor = 50.0;
    };

  public:
    FireflyFilterNode(const SharedContext context,
             const ResourceAllocatorHandle allocator);

    ~FireflyFilterNode();

    std::string name() override {
        return "Firefly Filter";
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

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;

    static constexpr uint32_t quartile_local_size_x = 8;
    static constexpr uint32_t quartile_local_size_y = 8;
    static constexpr uint32_t filter_local_size_x = 16;
    static constexpr uint32_t filter_local_size_y = 16;

    vk::ImageCreateInfo irr_create_info;

    uint32_t quartile_group_count_x;
    uint32_t quartile_group_count_y;
    uint32_t filter_group_count_x;
    uint32_t filter_group_count_y;

    TextureHandle quartile_texture;

    ShaderModuleHandle quartile_module;
    ShaderModuleHandle filter_module;

    FilterPushConstant filter_pc;
    QuartilePushConstant quartile_pc;

    PipelineHandle quartile;
    PipelineHandle filter;

    std::vector<TextureHandle> graph_textures;
    std::vector<DescriptorSetHandle> graph_sets;
    DescriptorSetLayoutHandle graph_layout;
    DescriptorPoolHandle graph_pool;

    DescriptorSetLayoutHandle quartile_desc_layout;
    DescriptorPoolHandle quartile_desc_pool;
    DescriptorSetLayoutHandle filter_desc_layout;
    DescriptorPoolHandle filter_desc_pool;
    DescriptorSetHandle quartile_set;
    DescriptorSetHandle filter_set;
};

} // namespace merian
