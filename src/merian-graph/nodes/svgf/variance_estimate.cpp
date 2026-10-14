#include "merian-graph/nodes/svgf/variance_estimate.hpp"
#include "merian-graph/nodes/svgf/config.h"

#include "merian/shader/shader_object.hpp"
#include "merian/utils/math.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/math.hpp"

namespace merian {

namespace {
constexpr const char* MODULE = "merian-graph/nodes/svgf/svgf_variance_estimate.slang";
}

DeviceSupportInfo
SVGFVarianceEstimate::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return svgf_device_support(query_info, MODULE);
}

std::vector<InputConnectorDescriptor> SVGFVarianceEstimate::describe_inputs() {
    return {
        {.name = "src", .connector = con_src, .access = ConnectorAccess::compute_read},
        {.name = "history", .connector = con_history, .access = ConnectorAccess::compute_read},
        {.name = "gbuffer", .connector = con_gbuffer, .access = ConnectorAccess::compute_read},
    };
}

std::vector<OutputConnectorDescriptor>
SVGFVarianceEstimate::describe_outputs(const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo src_create_info = io_layout[con_src]->get_create_info_or_throw();
    const vk::Format out_format =
        format != vk::Format::eUndefined ? format : src_create_info.format;
    extent = src_create_info.extent;
    if (options.kaleidoscope) {
        const uint32_t padding_multiple =
            std::max(1u << (std::max(filter_levels, 1u) - 1),
                     svgf_filter_local_size(io_layout.get_context(), options));
        extent.width = round_up(extent.width, padding_multiple);
        extent.height = round_up(extent.height, padding_multiple);
    }

    con_out = ManagedVkImageOut::create(out_format, extent);
    // the packed gbuffer of the kaleidoscope filter
    con_gbuf =
        ManagedVkImageOut::create(vk::Format::eR32G32B32A32Uint,
                                  options.kaleidoscope ? extent : vk::Extent3D{1, 1, extent.depth});
    return {
        {.name = "out", .connector = con_out, .access = ConnectorAccess::compute_write},
        {.name = "gbuf", .connector = con_gbuf, .access = ConnectorAccess::compute_write},
    };
}

SVGFVarianceEstimate::NodeStatusFlags
SVGFVarianceEstimate::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                   const NodeIO& io,
                                   [[maybe_unused]] const NodeConnectionInfo& info,
                                   [[maybe_unused]] Submission& submission) {
    const ContextHandle& context = io.get_context();
    local_size = workgroup_size_for_shared_memory_with_halo(context, VE_SHARED_MEMORY_PER_PIXEL,
                                                            SVGF_VE_HALO_RADIUS, 32, 16);
    entry_point = svgf_entry_point(context->get_shader_compile_context(),
                                   io[con_gbuffer]->get_layout()->get_composition(), MODULE,
                                   options, con_out->get_create_info()->format);

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size, local_size, filter_levels);
    pipeline = ComputePipeline::create(entry_point->get_pipeline_layout(context),
                                       entry_point->specialize(spec_builder.build()));
    globals = entry_point->create_global_shader_object(context, io.get_allocator());
    return {};
}

SVGFVarianceEstimate::NodeStatusFlags SVGFVarianceEstimate::process(const NodeIO& io,
                                                                    const NodeProcessInfo& info,
                                                                    Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    io.bind(globals);

    cmd->bind(pipeline);
    entry_point->bind_global(globals, cmd, pipeline, info.get_shader_object_allocator());
    PushConstant precomputed_pc = pc;
    precomputed_pc.depth_accept = -10.0f / precomputed_pc.depth_accept;
    cmd->push_constant(pipeline, precomputed_pc);
    cmd->dispatch(extent, local_size, local_size);
    return {};
}

SVGFVarianceEstimate::NodeStatusFlags SVGFVarianceEstimate::properties(Properties& config) {
    config.config_float("spatial falloff", pc.spatial_falloff,
                        "higher means only use spatial with very low history", 0.01);
    config.config_float("spatial bias", pc.spatial_bias,
                        "higher means use spatial information longer before using the falloff",
                        0.1);
    float angle = std::acos(pc.normal_reject_cos);
    config.config_angle("normal reject", angle, "Reject points with farther apart", 0, 180);
    pc.normal_reject_cos = std::cos(angle);
    config.config_float("depth accept", pc.depth_accept, "More means more reuse");
    config.output_text("local size: {}", local_size);
    return {};
}

void SVGFVarianceEstimate::set_options(const SVGFOptions& options) {
    this->options = options;
}

void SVGFVarianceEstimate::set_filter_levels(const uint32_t filter_levels) {
    this->filter_levels = filter_levels;
}

void SVGFVarianceEstimate::set_format(const vk::Format format) {
    this->format = format;
}

} // namespace merian
