#pragma once

#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/graph/node.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class SVGFNode : public Node {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct VarianceEstimatePushConstant {
        int fresh_threshold = 4;
        float fresh_variance_boost = 4.0;
    };

    struct FilterPushConstant {
        int gap;
        float param_z = 10;  // parameter for depth      = 1   larger blurs more 
        float param_n = 64;  // parameter for normals    = 128 smaller blurs more
        float param_l = 8;   // parameter for brightness = 4   larger blurs more
    };

    struct TAAPushConstant {
        float blend_alpha = 0.9;
        float rejection_threshold = 1.0;
        int show_variance_estimate = 0;
    };

  public:
    SVGFNode(const SharedContext context, const ResourceAllocatorHandle allocator);

    ~SVGFNode();

    std::string name() override {
        return "SVGF";
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

    void get_configuration(Configuration& config) override;

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;

    ShaderModuleHandle variance_estimate_module;
    ShaderModuleHandle filter_module;
    ShaderModuleHandle taa_module;

    VarianceEstimatePushConstant variance_estimate_pc;
    FilterPushConstant filter_pc;
    TAAPushConstant taa_pc;

    vk::ImageCreateInfo irr_create_info;

    PipelineHandle variance_estimate;
    PipelineHandle filter;
    PipelineHandle taa;

    uint32_t group_size_x;
    uint32_t group_size_y;

    int svgf_iterations = 4;

    std::vector<TextureHandle> graph_textures;
    std::vector<DescriptorSetHandle> graph_sets;
    DescriptorSetLayoutHandle graph_layout;
    DescriptorPoolHandle graph_pool;

    DescriptorSetLayoutHandle ping_pong_layout;
    DescriptorPoolHandle filter_pool;
    struct EAWRes {
        TextureHandle ping_pong;
        // Set reads from this resources and writes to i ^ 1
        DescriptorSetHandle set;
    };
    std::array<EAWRes, 2> ping_pong_res; // Ping pong sets
};

} // namespace merian
