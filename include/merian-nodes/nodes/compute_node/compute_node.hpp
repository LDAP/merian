#pragma once

#include "merian-nodes/graph/node.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/entry_point.hpp"

#include <optional>

namespace merian_nodes {

// A general purpose compute node.
// The pipeline is automatically rebuild if ShaderModule or SpecializationInfo pointer change.
class AbstractCompute : public Node {

  public:
    AbstractCompute(const ContextHandle& context,
                    const std::optional<uint32_t> push_constant_size = std::nullopt);

    virtual ~AbstractCompute() {}

    // Return a pointer to your push constant if push_constant_size is not std::nullop
    // In every run (rebuilds the pipeline if handle changed.)
    virtual const void* get_push_constant([[maybe_unused]] GraphRun& run,
                                          [[maybe_unused]] const NodeIO& io) {
        throw std::runtime_error{
            "get_push_constant must be overwritten when push_constant_size is not std::nullopt"};
    }

    // Return the group count for x, y and z
    // Called in every run
    virtual std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept = 0;

    // In every run (rebuilds the pipeline if handle changed.)
    virtual EntryPointHandle get_entry_point() = 0;

    virtual NodeStatusFlags
    on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                 const DescriptorSetLayoutHandle& descriptor_set_layout) override final;

    virtual void process(GraphRun& run,
                         const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) override final;

  protected:
    const ContextHandle context;
    const std::optional<uint32_t> push_constant_size;

  private:
    EntryPointHandle current_shader_module;

    DescriptorSetLayoutHandle descriptor_set_layout;
    PipelineHandle pipe;
};

template <class PushConstant> class TypedPCAbstractCompute : public AbstractCompute {
  public:
    TypedPCAbstractCompute(const ContextHandle& context)
        : AbstractCompute(context, sizeof(PushConstant)) {}

    virtual const void* get_push_constant([[maybe_unused]] GraphRun& run,
                                          [[maybe_unused]] const NodeIO& io) final {
        return &get_typed_push_constant(run, io);
    }

    virtual const PushConstant& get_typed_push_constant([[maybe_unused]] GraphRun& run,
                                                        [[maybe_unused]] const NodeIO& io) = 0;
};

} // namespace merian_nodes
