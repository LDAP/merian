#pragma once

#include "merian-nodes/graph/node.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/shader/shader_module.hpp"

#include <optional>

namespace merian_nodes {

// A general purpose compute node.
// The graph resources are bound in set 0 and order input images, input buffers, output images,
// output buffers. Input images are bound as sampler2d, output images as image2d.
//
// A rebuild regenerates the pipeline, this allows an update to spec constants.
class ComputeNode : public Node {

  public:
    ComputeNode(const SharedContext context,
                const std::string& name,
                const std::optional<uint32_t> push_constant_size = std::nullopt);

    virtual ~ComputeNode() {}

    // Return a SpecializationInfoHandle if you want to add specialization constants
    // Called at the first build
    virtual SpecializationInfoHandle get_specialization_info() const noexcept {
        return MERIAN_SPECIALIZATION_INFO_NONE;
    }

    // Return a pointer to your push constant if push_constant_size is not std::nullop
    // Called in every run
    virtual const void* get_push_constant([[maybe_unused]] GraphRun& run) {
        throw std::runtime_error{
            "get_push_constant must be overwritten when push_constant_size is not std::nullopt"};
    }

    // Return the group count for x,y and z
    // Called in every run
    virtual std::tuple<uint32_t, uint32_t, uint32_t> get_group_count() const noexcept = 0;

    // Called at the first build
    virtual ShaderModuleHandle get_shader_module() = 0;

    virtual NodeStatusFlags
    on_connected(const DescriptorSetLayoutHandle& descriptor_set_layout) override final;

    virtual void process(GraphRun& run,
                         const vk::CommandBuffer& cmd,
                         const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) override final;

  protected:
    const SharedContext context;
    const std::optional<uint32_t> push_constant_size;

  private:
    PipelineHandle pipe;
    PipelineLayoutHandle pipe_layout;
};

} // namespace merian_nodes
