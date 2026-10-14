#include "merian-graph/nodes/svgf/filter.hpp"

#include "merian/shader/shader_object.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

namespace {
constexpr const char* MODULE = "merian-graph/nodes/svgf/svgf_filter.slang";
}

DeviceSupportInfo SVGFFilter::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return svgf_device_support(query_info, MODULE);
}

std::vector<InputConnectorDescriptor> SVGFFilter::describe_inputs() {
    return {
        {.name = "src", .connector = con_src, .access = ConnectorAccess::compute_read},
        {.name = "gbuf_src", .connector = con_gbuf_src, .access = ConnectorAccess::compute_read},
        {.name = "gbuffer", .connector = con_gbuffer, .access = ConnectorAccess::compute_read},
    };
}

std::vector<OutputConnectorDescriptor> SVGFFilter::describe_outputs(const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo src = io_layout[con_src]->get_create_info_or_throw();
    const vk::ImageCreateInfo gbuf_src = io_layout[con_gbuf_src]->get_create_info_or_throw();
    con_out = ManagedVkImageOut::create(src.format, src.extent);
    con_gbuf = ManagedVkImageOut::create(gbuf_src.format, gbuf_src.extent);
    return {
        {.name = "out", .connector = con_out, .access = ConnectorAccess::compute_write},
        {.name = "gbuf", .connector = con_gbuf, .access = ConnectorAccess::compute_write},
    };
}

SVGFFilter::NodeStatusFlags
SVGFFilter::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                         const NodeIO& io,
                         [[maybe_unused]] const NodeConnectionInfo& info,
                         [[maybe_unused]] Submission& submission) {
    const ContextHandle& context = io.get_context();
    local_size = svgf_filter_local_size(context, options);
    entry_point = svgf_entry_point(context->get_shader_compile_context(),
                                   io[con_gbuffer]->get_layout()->get_composition(), MODULE,
                                   options, con_out->get_create_info()->format);

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size, local_size, 1 << level, level, level_count - 1);
    pipeline = ComputePipeline::create(entry_point->get_pipeline_layout(context),
                                       entry_point->specialize(spec_builder.build()));
    globals = entry_point->create_global_shader_object(context, io.get_allocator());
    return {};
}

SVGFFilter::NodeStatusFlags
SVGFFilter::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    io.bind(globals);

    cmd->bind(pipeline);
    entry_point->bind_global(globals, cmd, pipeline, info.get_shader_object_allocator());
    SVGFFilterParameters precomputed = parameters;
    precomputed.depth = -10.0f / precomputed.depth;
    cmd->push_constant(pipeline, precomputed);
    cmd->dispatch(con_out->get_create_info()->extent, local_size, local_size);
    return {};
}

SVGFFilter::NodeStatusFlags SVGFFilter::properties(Properties& config) {
    parameters.properties(config);
    return {};
}

uint32_t SVGFFilter::get_local_size() const {
    return local_size;
}

void SVGFFilter::set_options(const SVGFOptions& options) {
    this->options = options;
}

void SVGFFilter::set_level(const uint32_t level, const uint32_t level_count) {
    this->level = level;
    this->level_count = level_count;
}

void SVGFFilter::set_parameters(const SVGFFilterParameters& parameters) {
    this->parameters = parameters;
}

} // namespace merian
