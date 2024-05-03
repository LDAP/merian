#pragma once

#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class MedianApproxNode : public Node {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        float min = 0;
        float max = 1000;
    };

  public:
    MedianApproxNode(const SharedContext context,
                     const ResourceAllocatorHandle allocator,
                     const int component = 0);

    virtual ~MedianApproxNode();

    virtual std::string name() override;

    virtual std::tuple<std::vector<NodeInputDescriptorImage>,
                       std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    virtual std::tuple<std::vector<NodeOutputDescriptorImage>,
                       std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(
        const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
        const std::vector<NodeOutputDescriptorBuffer>& connected_buffer_outputs) override;

    virtual void cmd_build(const vk::CommandBuffer& cmd, const std::vector<NodeIO>& ios) override;

    virtual void cmd_process(const vk::CommandBuffer& cmd,
                             GraphRun& run,
                             const std::shared_ptr<FrameData>& frame_data,
                             const uint32_t set_index,
                             const NodeIO& io) override;

    virtual void get_configuration(Configuration& config, bool& needs_rebuild) override;

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;
    const int component;

    PushConstant pc;

    std::vector<TextureHandle> graph_textures;
    std::vector<DescriptorSetHandle> graph_sets;
    DescriptorSetLayoutHandle graph_layout;
    DescriptorPoolHandle graph_pool;

    ShaderModuleHandle histogram;
    ShaderModuleHandle reduce;

    PipelineHandle pipe_histogram;
    PipelineHandle pipe_reduce;
};

} // namespace merian
