#include "compute_node.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"

namespace merian_nodes {

AbstractCompute::AbstractCompute(const SharedContext context,
                                 const std::optional<uint32_t> push_constant_size)
    : Node(), context(context), push_constant_size(push_constant_size) {}

AbstractCompute::NodeStatusFlags
AbstractCompute::on_connected(const DescriptorSetLayoutHandle& descriptor_set_layout) {
    this->descriptor_set_layout = descriptor_set_layout;
    this->pipe.reset();

    return {};
}

void AbstractCompute::process(GraphRun& run,
                              const vk::CommandBuffer& cmd,
                              const DescriptorSetHandle& descriptor_set,
                              const NodeIO& io) {
    PipelineHandle& old_pipeline = io.frame_data<PipelineHandle>();
    old_pipeline.reset();

    const auto spec_info = get_specialization_info();
    const auto shader = get_shader_module();

    if (spec_info && shader &&
        (!pipe || current_spec_info != spec_info || current_shader_module != shader)) {
        SPDLOG_DEBUG("(re)create pipeline");
        old_pipeline = pipe;

        auto pipe_builder = PipelineLayoutBuilder(context);
        if (push_constant_size.has_value()) {
            pipe_builder.add_push_constant(push_constant_size.value());
        }

        PipelineLayoutHandle pipe_layout =
            pipe_builder.add_descriptor_set_layout(descriptor_set_layout).build_pipeline_layout();
        pipe = std::make_shared<ComputePipeline>(pipe_layout, get_shader_module(),
                                                 get_specialization_info());

        current_spec_info = get_specialization_info();
        current_shader_module = get_shader_module();
    }

    if (pipe) {
        pipe->bind(cmd);
        pipe->bind_descriptor_set(cmd, descriptor_set);
        if (push_constant_size.has_value())
            pipe->push_constant(cmd, get_push_constant(run));
        auto [x, y, z] = get_group_count();
        cmd.dispatch(x, y, z);
    }
}

} // namespace merian_nodes
