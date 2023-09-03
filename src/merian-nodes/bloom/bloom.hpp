#pragma once

#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class BloomNode : public Node {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        float threshold = 10.0;
        float strength = 0.001;
    };

  public:
    BloomNode(const SharedContext context, const ResourceAllocatorHandle allocator);

    virtual ~BloomNode();

    virtual std::string name() override;

    virtual std::tuple<std::vector<NodeInputDescriptorImage>,
                       std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    virtual std::tuple<std::vector<NodeOutputDescriptorImage>,
                       std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(
        const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
        const std::vector<NodeOutputDescriptorBuffer>& connected_buffer_outputs) override;

    virtual void cmd_build(const vk::CommandBuffer& cmd,
                           const std::vector<std::vector<ImageHandle>>& image_inputs,
                           const std::vector<std::vector<BufferHandle>>& buffer_inputs,
                           const std::vector<std::vector<ImageHandle>>& image_outputs,
                           const std::vector<std::vector<BufferHandle>>& buffer_outputs) override;

    virtual void cmd_process(const vk::CommandBuffer& cmd,
                             GraphRun& run,
                             const uint32_t set_index,
                             const std::vector<ImageHandle>& image_inputs,
                             const std::vector<BufferHandle>& buffer_inputs,
                             const std::vector<ImageHandle>& image_outputs,
                             const std::vector<BufferHandle>& buffer_outputs) override;

    virtual void get_configuration(Configuration& config, bool& needs_rebuild) override;

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;

    PushConstant pc;

    std::vector<TextureHandle> graph_textures;
    std::vector<DescriptorSetHandle> graph_sets;
    DescriptorSetLayoutHandle graph_layout;
    DescriptorPoolHandle graph_pool;

    ShaderModuleHandle separate_module;
    ShaderModuleHandle composite_module;

    PipelineHandle separate;
    PipelineHandle composite;

    int32_t mode = 0;
};

} // namespace merian
