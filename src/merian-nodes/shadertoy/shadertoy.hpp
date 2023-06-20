#pragma once

#include "glm/ext/vector_float2.hpp"
#include "merian/io/file_loader.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class ShadertoyNode : public merian::Node {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

  private:
    struct PushConstant {
        glm::vec2 iResolution{};
        float iTime{};
        float iTimeDelta{};
        float iFrame{};
    };

  public:
    ShadertoyNode(const SharedContext context,
                  const ResourceAllocatorHandle alloc,
                  std::string path,
                  FileLoader loader,
                  uint32_t width = 1920,
                  uint32_t height = 1080);

    std::string name() override {
        return "ShadertoyNode";
    }

    void set_resolution(uint32_t width = 1920, uint32_t height = 1080);

    // Called everytime before the graph is run. Can be used to request a rebuild for example.
    virtual void pre_process(NodeStatus& status) override;

    virtual std::tuple<std::vector<merian::NodeInputDescriptorImage>,
                       std::vector<merian::NodeInputDescriptorBuffer>>
    describe_inputs() override;

    virtual std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
                       std::vector<merian::NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<merian::NodeOutputDescriptorImage>&,
                     const std::vector<merian::NodeOutputDescriptorBuffer>&) override;

    virtual void cmd_build(const vk::CommandBuffer&,
                           const std::vector<std::vector<merian::ImageHandle>>&,
                           const std::vector<std::vector<merian::BufferHandle>>&,
                           const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                           const std::vector<std::vector<merian::BufferHandle>>&) override;

    virtual void cmd_process(const vk::CommandBuffer& cmd,
                             const uint64_t iteration,
                             const uint32_t set_index,
                             const std::vector<merian::ImageHandle>&,
                             const std::vector<merian::BufferHandle>&,
                             const std::vector<merian::ImageHandle>&,
                             const std::vector<merian::BufferHandle>&) override;

  private:
    const SharedContext context;
    const ResourceAllocatorHandle alloc;

    uint32_t width;
    uint32_t height;

    DescriptorSetLayoutHandle layout;
    DescriptorPoolHandle pool;
    std::vector<DescriptorSetHandle> sets;
    std::vector<TextureHandle> textures;
    PipelineHandle pipe;

    PushConstant constant;
    Stopwatch sw;
    bool requires_rebuild = true;
};

} // namespace merian
