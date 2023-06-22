#pragma once

#include "glm/ext/vector_float2.hpp"
#include "merian-nodes/compute_node/compute_node.hpp"
#include "merian/io/file_loader.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

// A generator node that pushes the Shadertoy variables as push constant.
class ShadertoyNode : public ComputeNode {
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
                  const std::string& path,
                  FileLoader loader,
                  const uint32_t width = 1920,
                  const uint32_t height = 1080);

    ShadertoyNode(const SharedContext context,
                  const ResourceAllocatorHandle alloc,
                  const std::size_t spv_size,
                  const uint32_t spv[],
                  const uint32_t width = 1920,
                  const uint32_t height = 1080);

    virtual std::string name() override {
        return "ShadertoyNode";
    }

    void set_resolution(uint32_t width, uint32_t height);

    void pre_process(NodeStatus& status) override final;

    std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
                       std::vector<merian::NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<merian::NodeOutputDescriptorImage>&,
                     const std::vector<merian::NodeOutputDescriptorBuffer>&) override final;

    SpecializationInfoHandle get_specialization_info() const noexcept override final;

    const void* get_push_constant() override final;

    std::tuple<uint32_t, uint32_t, uint32_t> get_group_count() const noexcept override final;

    ShaderModuleHandle get_shader_module() override final;

  private:
    uint32_t width;
    uint32_t height;

    ShaderModuleHandle shader;

    PushConstant constant;
    Stopwatch sw;
    bool requires_rebuild = false;
};

} // namespace merian
