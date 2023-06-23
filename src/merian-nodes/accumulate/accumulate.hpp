#pragma once

#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

namespace merian {

// An accumulate node for vk::Format::eR32G32B32A32Sfloat images
class AccumulateF32ImageNode : public Node {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

  public:
    AccumulateF32ImageNode(const SharedContext context, const ResourceAllocatorHandle alloc);

    std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
               std::vector<merian::NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<merian::NodeOutputDescriptorImage>& connected_image_outputs,
                     const std::vector<merian::NodeOutputDescriptorBuffer>&) override;

    void cmd_build(const vk::CommandBuffer&,
                   const std::vector<std::vector<merian::ImageHandle>>& image_inputs,
                   const std::vector<std::vector<merian::BufferHandle>>&,
                   const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                   const std::vector<std::vector<merian::BufferHandle>>&) override;

    void cmd_process(const vk::CommandBuffer& cmd,
                     GraphRun& run,
                     const uint32_t set_index,
                     const std::vector<merian::ImageHandle>&,
                     const std::vector<merian::BufferHandle>&,
                     const std::vector<merian::ImageHandle>&,
                     const std::vector<merian::BufferHandle>&) override;

  private:
    const SharedContext context;
    const ResourceAllocatorHandle alloc;

    DescriptorSetLayoutHandle layout;
    DescriptorPoolHandle pool;
    std::vector<DescriptorSetHandle> sets;
    std::vector<TextureHandle> in_textures;
    std::vector<TextureHandle> out_textures;
    PipelineHandle pipe;

    uint32_t group_count_x;
    uint32_t group_count_y;
};

} // namespace merian
