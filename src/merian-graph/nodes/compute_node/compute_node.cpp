#include "merian-graph/nodes/compute_node/compute_node.hpp"

#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_global_session.hpp"

namespace merian {

AbstractCompute::AbstractCompute(const std::optional<uint32_t> push_constant_size)
    : push_constant_size(push_constant_size) {}

void AbstractCompute::invalidate_shader() {
    kernel->invalidate();
    bump_slang_source_epoch();
}

void AbstractCompute::initialize(const ContextHandle& context,
                                 const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    kernel.emplace(
        context, allocator, compile_context, [this] { return create_composition(); }, spec_info);
}

AbstractCompute::NodeStatusFlags AbstractCompute::on_connected(const NodeConnectedInfo& info) {
    const NodeIOLayout& io_layout = info.io_layout;
    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            kernel->reload(std::any_cast<bool>(force), compile_context);
            return true;
        });

    return {};
}

void AbstractCompute::process(GraphRun& run, const NodeIO& io) {
    ShaderCursor cursor = kernel->globals_cursor();
    write_constants(run, io, cursor);

    const PipelineHandle pipe = kernel->bind(run, io);

    const CommandBufferHandle& cmd = run.get_cmd();
    if (push_constant_size.has_value()) {
        cmd->push_constant(pipe, get_push_constant(run, io));
    }
    const auto [x, y, z] = get_group_count(io);
    cmd->dispatch(x, y, z);
}

} // namespace merian
