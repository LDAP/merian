#include "merian-graph/nodes/accumulate/accumulate.hpp"

#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"

namespace merian {

namespace {
constexpr const char* ACCUMULATE_MODULE = "merian-graph/nodes/accumulate/accumulate.slang";
constexpr const char* PERCENTILES_MODULE =
    "merian-graph/nodes/accumulate/calculate_percentiles.slang";
} // namespace

Accumulate::Accumulate() {}

Accumulate::~Accumulate() {}

DeviceSupportInfo Accumulate::query_device_support(const DeviceSupportQueryInfo& query_info) {
    DeviceSupportInfo support{true};
    for (const char* module : {ACCUMULATE_MODULE, PERCENTILES_MODULE}) {
        const auto composition = SlangComposition::create();
        composition->add_module_from_path(module, true);
        support = support & SlangProgram::create(query_info.compile_context, composition)
                                .get()
                                ->query_device_support(query_info);
    }
    return support;
}

void Accumulate::initialize(const ContextHandle& context,
                            const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    percentile_kernel.emplace(context, allocator, compile_context, PERCENTILES_MODULE,
                              percentile_spec_info);
    accumulate_kernel.emplace(context, allocator, compile_context, ACCUMULATE_MODULE,
                              accumulate_spec_info);
}

std::vector<InputConnectorDescriptor> Accumulate::describe_inputs() {
    return {
        {"src", con_src, ConnectorAccess::compute_read},
        {"gbuffer", con_gbuffer, ConnectorAccess::compute_read},
        {"prev_out", con_prev_out, ConnectorAccess::compute_read, 1},
        {"prev_gbuffer", con_prev_gbuffer, ConnectorAccess::compute_read, 1},
        {"prev_history", con_prev_history, ConnectorAccess::compute_read, 1},
    };
}

std::vector<OutputConnectorDescriptor> Accumulate::describe_outputs(const NodeIOLayout& io_layout) {

    irr_create_info = io_layout[con_src]->get_create_info_or_throw();
    con_out =
        ManagedVkImageOut::create(format.value_or(irr_create_info.format), irr_create_info.extent);
    con_history = ManagedVkImageOut::create(vk::Format::eR32Sfloat, irr_create_info.extent);

    io_layout.register_event_listener(clear_event_listener_pattern,
                                      [this](const GraphEvent::Info&, const GraphEvent::Data&) {
                                          request_clear();
                                          return true;
                                      });

    return {
        {"out", con_out, ConnectorAccess::compute_write},
        {"history", con_history, ConnectorAccess::compute_write},

    };
}

Accumulate::NodeStatusFlags Accumulate::on_connected(const NodeConnectedInfo& info) {
    const NodeIOLayout& io_layout = info.io_layout;
    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            for (auto* kernel : {&percentile_kernel, &accumulate_kernel}) {
                (*kernel)->reload(std::any_cast<bool>(force), compile_context);
            }
            return true;
        });

    percentile_group_count_x =
        (irr_create_info.extent.width + PERCENTILE_LOCAL_SIZE_X - 1) / PERCENTILE_LOCAL_SIZE_X;
    percentile_group_count_y =
        (irr_create_info.extent.height + PERCENTILE_LOCAL_SIZE_Y - 1) / PERCENTILE_LOCAL_SIZE_Y;
    filter_group_count_x =
        (irr_create_info.extent.width + FILTER_LOCAL_SIZE_X - 1) / FILTER_LOCAL_SIZE_X;
    filter_group_count_y =
        (irr_create_info.extent.height + FILTER_LOCAL_SIZE_Y - 1) / FILTER_LOCAL_SIZE_Y;

    vk::ImageCreateInfo quartile_image_create_info = irr_create_info;
    quartile_image_create_info.format = vk::Format::eR32G32B32A32Sfloat;
    quartile_image_create_info.usage |=
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
    quartile_image_create_info.setExtent({percentile_group_count_x, percentile_group_count_y, 1});
    const ImageHandle quartile_image = allocator->create_image(
        quartile_image_create_info, MemoryMappingType::NONE, "accum node, quartiles");
    vk::ImageViewCreateInfo quartile_image_view_create_info{
        {}, *quartile_image,        vk::ImageViewType::e2D, quartile_image->get_format(),
        {}, first_level_and_layer()};
    percentile_texture =
        allocator->create_texture(quartile_image, quartile_image_view_create_info,
                                  allocator->get_sampler_pool()->linear_mirrored_repeat());

    auto quartile_spec_builder = SpecializationInfoBuilder();
    quartile_spec_builder.add_entry(PERCENTILE_LOCAL_SIZE_X, PERCENTILE_LOCAL_SIZE_Y);
    percentile_spec_info.set(quartile_spec_builder.build());

    auto accum_spec_builder = SpecializationInfoBuilder();
    const uint32_t wg_rounded_irr_size_x = percentile_group_count_x * PERCENTILE_LOCAL_SIZE_X;
    const uint32_t wg_rounded_irr_size_y = percentile_group_count_y * PERCENTILE_LOCAL_SIZE_Y;
    accum_spec_builder.add_entry(FILTER_LOCAL_SIZE_X, FILTER_LOCAL_SIZE_Y, wg_rounded_irr_size_x,
                                 wg_rounded_irr_size_y, filter_mode, extended_search, reuse_border,
                                 enable_mv, gbuffer_check_mode);
    accumulate_spec_info.set(accum_spec_builder.build());

    return {};
}

void Accumulate::process(GraphRun& run, const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();
    accumulate_pc.iteration = run.get_total_iteration();

    if (accumulate_pc.firefly_filter_enable == VK_TRUE ||
        accumulate_pc.adaptive_alpha_reduction > 0.0f) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "compute percentiles");
        const auto bar = percentile_texture->get_image()->barrier(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead,
            vk::AccessFlagBits::eShaderWrite, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers(), true);
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader, bar);

        percentile_kernel->globals_cursor()["quartiles"].write(percentile_texture->get_view(),
                                                               vk::ImageLayout::eGeneral);
        const auto pipe = percentile_kernel->bind(run, io);
        cmd->push_constant(pipe, percentile_pc);
        cmd->dispatch(percentile_group_count_x, percentile_group_count_y, 1);
    }

    auto bar = percentile_texture->get_image()->barrier(
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        all_levels_and_layers());
    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                 vk::PipelineStageFlagBits::eComputeShader, bar);

    {
        if (run.get_iteration() == 0 || clear) {
            accumulate_pc.clear = VK_TRUE;
            io.send_event("clear");
            clear = false;
        } else {
            accumulate_pc.clear = VK_FALSE;
        }

        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "accumulate");
        accumulate_kernel->globals_cursor()["quartiles"].write(
            percentile_texture, vk::ImageLayout::eShaderReadOnlyOptimal);
        const auto pipe = accumulate_kernel->bind(run, io);
        cmd->push_constant(pipe, accumulate_pc);
        cmd->dispatch(filter_group_count_x, filter_group_count_y);
    }
}

Accumulate::NodeStatusFlags Accumulate::properties(Properties& config) {
    bool needs_rebuild = false;
    config.st_separate("Accumulation");
    config.config_float("alpha", accumulate_pc.accum_alpha,
                        "Blend factor with the previous information. More means more reuse", 0.01f,
                        0.0f, 1.0f);
    config.config_float("max history", accumulate_pc.accum_max_hist,
                        "artificially limit the history counter. This can be a good alternative to "
                        "reducing the blend alpha");
    config.st_no_space();
    accumulate_pc.accum_max_hist =
        config.config_bool("inf history") ? INFINITY : accumulate_pc.accum_max_hist;
    clear |= config.config_bool("clear");
    config.st_no_space();
    needs_rebuild |= config.config_text(
        "clear event pattern", clear_event_listener_pattern, true,
        "Comma separated list of event patterns which trigger a clear. Press enter to confirm.");

    config.st_separate("Reproject");
    needs_rebuild |= config.config_bool("use motion vectors", enable_mv,
                                        "Reproject using motion vectors if connected.");
    needs_rebuild |= config.config_options(
        "gbuffer check", gbuffer_check_mode, {"always", "only if moving", "never"},
        Properties::OptionsStyle::DONT_CARE,
        "Validate reuse against the reprojected gbuffer (normal/depth).\n"
        "always: reject reprojections onto a mismatching surface.\n"
        "only if moving: accept where the motion vector is zero, validate otherwise.\n"
        "never: trust the motion vectors and skip the check (also disables extended search).");
    if (gbuffer_check_mode != 2) { // shown unless "never"
        float angle = std::acos(accumulate_pc.normal_reject_cos);
        config.config_angle("normal threshold", angle, "Reject points with normals farther apart",
                            0, 180);
        accumulate_pc.normal_reject_cos = std::cos(angle);
        config.config_percent("depth threshold", accumulate_pc.depth_reject_percent,
                              "Reject points with depths farther apart (relative to the max)");
    }
    needs_rebuild |=
        config.config_options("filter mode", filter_mode, {"nearest", "stochastic bilinear"});
    needs_rebuild |= config.config_bool(
        "extended search", extended_search,
        "search randomly in a 4x4 radius with weakened rejection thresholds for valid "
        "information if nothing was found.");
    needs_rebuild |=
        config.config_bool("reuse border", reuse_border,
                           "Reuse border information (if valid) for pixel where the motion vector "
                           "points outside of the image. Can lead to smearing.");
    config.st_separate("Firefly Suppression");
    config.config_bool("firefly filter enable", accumulate_pc.firefly_filter_enable);

    config.config_float("firefly filter bias", accumulate_pc.firefly_bias,
                        "Adds this value to the maximum allowed luminance.", 0.1);
    config.config_float("IPR factor", accumulate_pc.firefly_ipr_factor,
                        "Inter-percentile range factor. Increase to allow higher outliers.");
    config.st_separate();
    config.config_percent("firefly percentile lower", percentile_pc.firefly_percentile_lower);
    config.config_percent("firefly percentile upper", percentile_pc.firefly_percentile_upper);
    config.st_separate();
    config.config_float("hard clamp", accumulate_pc.firefly_hard_clamp, "DANGER: Introduces bias",
                        0.1);
    config.st_no_space();
    if (config.config_bool("inf clamp"))
        accumulate_pc.firefly_hard_clamp = INFINITY;

    config.st_separate("Adaptive alpha reduction");
    config.config_percent("adaptivity", accumulate_pc.adaptive_alpha_reduction,
                          "(1. - adaptivity) is the smallest factor that alpha is multipied with");
    config.config_float(
        "adaptivity IPR factor", accumulate_pc.adaptive_alpha_ipr_factor,
        "Inter-percentile range for adaptive reduction. Increase to soften reduction.", 0.1);
    config.st_separate();
    config.config_percent("adaptivity percentile lower",
                          percentile_pc.adaptive_alpha_percentile_lower);
    config.config_percent("adaptivity percentile upper",
                          percentile_pc.adaptive_alpha_percentile_upper);

    return needs_rebuild ? NodeStatusFlags{NEEDS_RECONNECT} : NodeStatusFlags{};
}

void Accumulate::request_clear() {
    clear = true;
}

} // namespace merian
