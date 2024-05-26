#pragma once

#include "merian-nodes/nodes/taa/config.h"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

namespace merian {

class TAANode : public ComputeNode {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        // higher value means more temporal reuse
        float temporal_alpha;
        int clamp_method;
    };

  public:
    TAANode(const SharedContext context,
            const ResourceAllocatorHandle allocator,
            const float alpha = 0.,
            const int clamp_method = MERIAN_NODES_TAA_CLAMP_MIN_MAX,
            const bool inverse_motion = false);

    std::string name() override {
        return "Temporal Anti-Aliasing";
    }

    std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(
        const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
        const std::vector<NodeOutputDescriptorBuffer>& connected_buffer_outputs) override;

    SpecializationInfoHandle get_specialization_info() const noexcept override;

    const void* get_push_constant([[maybe_unused]] GraphRun& run) override;

    std::tuple<uint32_t, uint32_t, uint32_t> get_group_count() const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    void get_configuration(Configuration& config, bool& needs_rebuild) override;

  private:
    const bool inverse_motion;
    ShaderModuleHandle shader;
    PushConstant pc;
    uint32_t width{};
    uint32_t height{};
};

} // namespace merian