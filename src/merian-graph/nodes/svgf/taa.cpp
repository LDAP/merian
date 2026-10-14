#include "merian-graph/nodes/svgf/taa.hpp"

#include "merian/shader/shader_object.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

namespace {
constexpr const char* MODULE = "merian-graph/nodes/svgf/svgf_taa.slang";

template <typename E>
bool config_options(Properties& config,
                    const std::string& label,
                    E& value,
                    const std::vector<std::string>& options,
                    const Properties::OptionsStyle style = Properties::OptionsStyle::DONT_CARE) {
    int index = static_cast<int>(value);
    if (config.config_options(label, index, options, style)) {
        value = static_cast<E>(index);
        return true;
    }
    return false;
}
} // namespace

DeviceSupportInfo SVGFTAA::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return svgf_device_support(query_info, MODULE);
}

std::vector<InputConnectorDescriptor> SVGFTAA::describe_inputs() {
    std::vector<GBufferGroup> gbuffer_groups = {{{GBufferField::MotionVectors}}};
    if (debug == Debug::NORMAL || debug == Debug::DEPTH || debug == Debug::GRAD_Z) {
        gbuffer_groups.push_back({{GBufferField::Normal, GBufferField::LinearZ, GBufferField::GradZ,
                                   GBufferField::DeltaZ}});
    }
    if (modulate_albedo || debug == Debug::ALBEDO) {
        gbuffer_groups.push_back({{GBufferField::Albedo}});
    }
    con_gbuffer = GBufferIn::create(gbuffer_groups);

    return {
        {.name = "src", .connector = con_src, .access = ConnectorAccess::compute_read},
        {.name = "prev_out",
         .connector = con_prev_out,
         .access = ConnectorAccess::compute_read,
         .delay = 1,
         .auto_connect = AutoConnect::output("out")},
        {.name = "gbuffer", .connector = con_gbuffer, .access = ConnectorAccess::compute_read},
        {.name = "mv",
         .connector = con_mv,
         .access = ConnectorAccess::compute_read,
         .optional = true},
    };
}

std::vector<OutputConnectorDescriptor> SVGFTAA::describe_outputs(const NodeIOLayout& io_layout) {
    // the source may be padded for the filter
    con_out = ManagedVkImageOut::create(io_layout[con_src]->get_create_info_or_throw().format,
                                        io_layout[con_gbuffer]->get_create_info().extent);
    return {{.name = "out", .connector = con_out, .access = ConnectorAccess::compute_write}};
}

SVGFTAA::NodeStatusFlags SVGFTAA::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                               const NodeIO& io,
                                               [[maybe_unused]] const NodeConnectionInfo& info,
                                               [[maybe_unused]] Submission& submission) {
    const ContextHandle& context = io.get_context();
    local_size = svgf_vendor_local_size(context);
    entry_point = svgf_entry_point(context->get_shader_compile_context(),
                                   io[con_gbuffer]->get_layout()->get_composition(), MODULE, {},
                                   con_out->get_create_info()->format);

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size, local_size, static_cast<int32_t>(debug),
                           static_cast<int32_t>(filter_prev), static_cast<int32_t>(clamping),
                           static_cast<int32_t>(mv_sampling), enable_mv, modulate_albedo,
                           static_cast<VkBool32>(io.is_connected(con_mv)));
    pipeline = ComputePipeline::create(entry_point->get_pipeline_layout(context),
                                       entry_point->specialize(spec_builder.build()));
    globals = entry_point->create_global_shader_object(context, io.get_allocator());
    return {};
}

SVGFTAA::NodeStatusFlags
SVGFTAA::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    io.bind(globals);

    cmd->bind(pipeline);
    entry_point->bind_global(globals, cmd, pipeline, info.get_shader_object_allocator());
    pc.reset = static_cast<VkBool32>(info.get_iteration() == 0);
    cmd->push_constant(pipeline, pc);
    cmd->dispatch(con_out->get_create_info()->extent, local_size, local_size);
    return {};
}

SVGFTAA::NodeStatusFlags SVGFTAA::properties(Properties& config) {
    bool needs_reconnect = false;
    config.config_float(
        "alpha", pc.blend_alpha,
        "Blend factor for the final image and the previous image. More means more reuse.", 0.01f,
        0.0f, 1.0f);
    needs_reconnect |=
        config.config_bool("enable motion vectors", enable_mv, "uses motion vectors if connected.");
    needs_reconnect |= config.config_bool(
        "modulate albedo", modulate_albedo,
        "Re-modulate the gbuffer albedo the renderer demodulated out. Disable if the renderer's "
        "'output/demodulate albedo' is off.");
    if (enable_mv) {
        needs_reconnect |=
            config_options(config, "mv sampling", mv_sampling, {"center", "magnitude dilation"},
                           Properties::OptionsStyle::COMBO);
    }
    needs_reconnect |= config_options(config, "filter", filter_prev, {"none", "catmull rom"},
                                      Properties::OptionsStyle::COMBO);
    needs_reconnect |= config_options(config, "clamping", clamping, {"min-max", "moments"},
                                      Properties::OptionsStyle::COMBO);
    if (clamping == Clamping::MOMENTS) {
        config.config_float(
            "rejection threshold", pc.rejection_threshold,
            "TAA rejection threshold for the previous frame, in units of standard deviation", 0.01);
    }
    needs_reconnect |= config_options(config, "debug", debug,
                                      {"none", "irradiance", "variance", "normal", "depth",
                                       "albedo", "grad z", "irradiance nan/inf", "mv"});
    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

bool SVGFTAA::modulates_albedo() const {
    return modulate_albedo;
}

} // namespace merian
