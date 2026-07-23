#pragma once

#include "merian-graph/graph/node.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"

#include "merian/vk/pipeline/specialization_info.hpp"

#include <optional>

namespace merian {

// A general purpose compute node backed by a runtime-compiled slang shader.
//
// Graph inputs and outputs are bound automatically to global shader variables named
// in_<connector> / out_<connector> (see NodeIO::bind). Per-run constants are written in
// write_constants; the pipeline is rebuilt when the specialization info changes.
class AbstractCompute : public Node {

  public:
    AbstractCompute(const std::optional<uint32_t> push_constant_size = std::nullopt);

    virtual ~AbstractCompute() = default;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    // Compose the node's shader. Called lazily on first process and after invalidate_shader().
    virtual SlangCompositionHandle create_composition() = 0;

    // Return a pointer to your push constant if push_constant_size is not std::nullopt.
    virtual const void* get_push_constant([[maybe_unused]] GraphRun& run,
                                          [[maybe_unused]] const NodeIO& io) {
        throw std::runtime_error{
            "get_push_constant must be overwritten when push_constant_size is not std::nullopt"};
    }

    // Write per-run constants into the globals cursor. Graph io fields are bound separately.
    virtual void write_constants([[maybe_unused]] GraphRun& run,
                                 [[maybe_unused]] const NodeIO& io,
                                 [[maybe_unused]] ShaderCursor& cursor) {}

    // Return the group count for x, y and z. Called in every run.
    virtual std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept = 0;

    virtual NodeStatusFlags on_connected(const NodeConnectedInfo& info) override;

    virtual void process(GraphRun& run, const NodeIO& io) override final;

  protected:
    // Force shader recompilation after the composed source changed. Retires the shared session so
    // a module recomposed under the same name is reloaded instead of served from the stale cache.
    void invalidate_shader();

    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;
    const std::optional<uint32_t> push_constant_size;

    // Pipeline is rebuilt when a new value is set.
    Versioned<SpecializationInfo> spec_info{MERIAN_SPECIALIZATION_INFO_NONE};

  private:
    std::optional<ComputeKernel> kernel;
};

template <class PushConstant> class TypedPCAbstractCompute : public AbstractCompute {
  public:
    TypedPCAbstractCompute() : AbstractCompute(sizeof(PushConstant)) {}

    virtual const void* get_push_constant([[maybe_unused]] GraphRun& run,
                                          [[maybe_unused]] const NodeIO& io) final {
        return &get_typed_push_constant(run, io);
    }

    virtual const PushConstant& get_typed_push_constant([[maybe_unused]] GraphRun& run,
                                                        [[maybe_unused]] const NodeIO& io) = 0;
};

} // namespace merian
