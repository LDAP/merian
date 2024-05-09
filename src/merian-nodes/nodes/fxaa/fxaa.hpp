#pragma once

#include "merian-nodes/nodes/compute_node/compute_node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

class FXAA : public ComputeNode {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        int32_t enable = 1;
    };

  public:
    FXAA(const SharedContext context,
         const ResourceAllocatorHandle allocator);

    std::string name() override {
        return "FXAA";
    }

    std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
               std::vector<merian::NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<merian::NodeOutputDescriptorImage>&,
                     const std::vector<merian::NodeOutputDescriptorBuffer>&) override;

    SpecializationInfoHandle get_specialization_info() const noexcept override;

    const void* get_push_constant([[maybe_unused]] GraphRun& run) override;

    std::tuple<uint32_t, uint32_t, uint32_t> get_group_count() const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    void get_configuration(Configuration& config, bool& needs_rebuild) override;

  private:
    vk::Extent3D extent;
    PushConstant pc;
};

} // namespace merian
