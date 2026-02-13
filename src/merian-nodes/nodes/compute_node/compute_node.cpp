#include "merian-nodes/nodes/compute_node/compute_node.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"

namespace merian {

AbstractCompute::AbstractCompute(const std::optional<uint32_t> push_constant_size)
    : push_constant_size(push_constant_size) {}

void AbstractCompute::initialize(const ContextHandle& context,
                                 const ResourceAllocatorHandle& /*allocator*/) {
    this->context = context;
}

AbstractCompute::NodeStatusFlags
AbstractCompute::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                              const DescriptorSetLayoutHandle& descriptor_set_layout) {
    this->descriptor_set_layout = descriptor_set_layout;
    this->pipe.reset();

    return {};
}

void AbstractCompute::process(GraphRun& run,
                              const DescriptorSetHandle& descriptor_set,
                              const NodeIO& io) {
    const auto shader = get_entry_point();

    if (shader && (!pipe || current_shader_module != shader)) {
        SPDLOG_DEBUG("(re)create pipeline");

        auto pipe_builder = PipelineLayoutBuilder(context);
        if (push_constant_size.has_value()) {
            pipe_builder.add_push_constant(push_constant_size.value());
        }

        PipelineLayoutHandle pipe_layout =
            pipe_builder.add_descriptor_set_layout(descriptor_set_layout).build_pipeline_layout();
        pipe = ComputePipeline::create(pipe_layout, shader);

        current_shader_module = shader;
    }

    if (pipe) {
        const CommandBufferHandle& cmd = run.get_cmd();
        cmd->bind(pipe);
        cmd->bind_descriptor_set(pipe, descriptor_set);
        if (push_constant_size.has_value())
            cmd->push_constant(pipe, get_push_constant(run, io));
        const auto [x, y, z] = get_group_count(io);
        cmd->dispatch(x, y, z);
    }
}

} // namespace merian
