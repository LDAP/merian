#include "merian-graph/nodes/gbuffer_rt/gbuffer.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace merian {

namespace {
// per axis, traced over native resolution
constexpr std::array<float, 6> RESOLUTION_SCALES = {1.f,       3.f / 4.f, 2.f / 3.f,
                                                    1.f / 2.f, 1.f / 3.f, 1.f / 4.f};
const std::vector<std::string> RESOLUTION_SCALE_NAMES = {"1", "3/4", "2/3", "1/2", "1/3", "1/4"};

vk::Extent3D scaled(const vk::Extent3D& extent, const float scale) {
    return {std::max(static_cast<uint32_t>(std::lround(extent.width * scale)), 1u),
            std::max(static_cast<uint32_t>(std::lround(extent.height * scale)), 1u), 1};
}
} // namespace

GBufferRTNode::GBufferRTNode() {}

DeviceSupportInfo GBufferRTNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_composition(GBufferLayout::complete()->get_composition());
    composition->add_module_from_path("merian-graph/nodes/gbuffer_rt/gbuffer_rt.slang", true);
    const auto program = SlangProgram::create(query_info.compile_context, composition);
    return DeviceSupportInfo::check(query_info, {"rayTracingPipeline", "accelerationStructure"},
                                    {"rayQuery"}) &
           program.get()->query_device_support(query_info);
}

void GBufferRTNode::initialize(const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();
}

std::vector<InputConnectorDescriptor> GBufferRTNode::describe_inputs() {
    return {{"scene", con_scene}};
}

std::vector<OutputConnectorDescriptor>
GBufferRTNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    con_gbuffer = GBufferOut::create(extent);
    con_emission = ManagedVkImageOut::create(emission_format, extent);

    return {
        {"gbuffer", con_gbuffer, ConnectorAccess::ray_tracing_write},
        {"emission", con_emission, ConnectorAccess::ray_tracing_write},
    };
}

GBufferRTNode::NodeStatusFlags
GBufferRTNode::on_connected(const NodeIOLayout& io_layout,
                            const NodeIO& io,
                            [[maybe_unused]] const NodeConnectionInfo& info,
                            [[maybe_unused]] Submission& submission) {

    // force the program graph to be rewired next process()
    composition = nullptr;

    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            if (composition) {
                if (std::any_cast<bool>(force)) {
                    composition->force_reload();
                } else {
                    composition->reload(compile_context->get_search_path_file_loader());
                }
            }
            return true;
        });

    emission_connected = io_layout.is_connected(con_emission);
    specular_hit_distance_requested =
        con_gbuffer->get_layout()->contains(GBufferField::SpecularHitDistance);
    gbuffer_composition = con_gbuffer->get_layout()->get_composition();

    if (const SceneHandle& scene = io[con_scene]; scene && scene->is_ready()) {
        ensure_pipeline(scene);
    }

    return {};
}

void GBufferRTNode::ensure_pipeline(const SceneHandle& scene) {
    if (composition) {
        return;
    }

    composition = SlangComposition::create();
    composition->add_composition(scene->get_composition());
    composition->add_composition(gbuffer_composition);
    composition->add_module_from_path("merian-graph/nodes/gbuffer_rt/gbuffer_rt.slang", true);
    update_gbuffer_constants();

    program = SlangProgram::create(compile_context, composition);
    entry_point = SlangProgramEntryPoint::create(program, "main");

    pipeline = Versioned<RayTracingPipeline>([this] {
        const auto ep = entry_point.get();
        return RayTracingPipelineBuilder()
            .add_raygen_group(ep->specialize())
            .build(ep->get_pipeline_layout(context));
    });
    pipeline.depends_on(entry_point);

    sbt = Versioned<ShaderBindingTable>(
        [this] { return ShaderBindingTable::create(pipeline.get(), resource_allocator); });
    sbt.depends_on(pipeline);

    globals_obj = Versioned<ShaderObject>(
        [this] { return entry_point->create_global_shader_object(context, resource_allocator); });
    globals_obj.depends_on(entry_point);
}

[[nodiscard]] GBufferRTNode::NodeStatusFlags
GBufferRTNode::pre_process(const NodeIO& io, [[maybe_unused]] const NodeProcessInfo& info) {
    const SceneHandle& scene = io[con_scene];
    const std::optional<vk::Extent3D> resolution = scene ? scene->get_resolution() : std::nullopt;
    resolution_from_scene = resolution.has_value();
    if (resolution) {
        native_extent = *resolution;
    }
    if (const vk::Extent3D traced = scaled(native_extent, RESOLUTION_SCALES[resolution_scale]);
        traced != extent) {
        extent = traced;
        return NodeStatusFlagBits::NEEDS_RECONNECT;
    }
    return {};
}

[[nodiscard]] GBufferRTNode::NodeStatusFlags
GBufferRTNode::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const auto& cmd = submission.get_cmd();
    const auto& scene = io[con_scene];

    ensure_pipeline(scene);

    const ShaderObjectAllocatorHandle& obj_allocator = info.get_shader_object_allocator();

    const auto ep = entry_point.get();
    const auto pipe = pipeline.get();
    const auto globals = globals_obj.get();

    if (emission_connected)
        globals->get_cursor()["emission"] = io[con_emission].get_texture();

    uint32_t mask = 0u;
    for (uint32_t bit = 0; bit < 8; ++bit) {
        if (mask_enabled[bit])
            mask |= (1u << bit);
    }
    globals->get_cursor()["params"]["instance_mask"] = mask;

    if (scene->is_ready()) {
        cmd->bind(pipe);
        ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
        ep->bind("gbuffer", io[con_gbuffer].w(), cmd, pipe, obj_allocator);
        ep->bind_global(globals, cmd, pipe, obj_allocator);
        cmd->trace_rays(sbt.get(), extent);
    }
    return {};
}

void GBufferRTNode::update_gbuffer_constants() {
    composition->add_module_from_string(
        "gbuffer_constants",
        fmt::format("namespace merian {{ export static const bool "
                    "merian_gbuffer_write_specular_hit_distance = {}; export static const int "
                    "merian_gbuffer_specular_lobe = {}; export static const float "
                    "merian_gbuffer_specular_max_roughness = {}; export static const int "
                    "merian_gbuffer_specular_max_bounces = {}; export static const bool "
                    "merian_gbuffer_specular_psr = {}; export static const bool "
                    "merian_gbuffer_write_emission = {}; export static const float "
                    "merian_gbuffer_texture_lod_scale = {:f}; }}",
                    specular_hit_distance_requested ? "true" : "false", specular_lobe,
                    specular_max_roughness, specular_max_bounces, specular_psr ? "true" : "false",
                    emission_connected ? "true" : "false", std::exp2(texture_lod_bias)));
}

GBufferRTNode::NodeStatusFlags GBufferRTNode::properties(Properties& config) {
    bool needs_reconnect = false;
    if (resolution_from_scene) {
        config.output_text("resolution from scene camera: {}x{}", native_extent.width,
                           native_extent.height);
    } else {
        needs_reconnect |= config.config_uint("width", &native_extent.width);
        needs_reconnect |= config.config_uint("height", &native_extent.height);
    }
    if (config.config_float("texture lod bias", texture_lod_bias,
                            "in mip levels; an upscaler asks for -1 on top of the scale", 0.05f)) {
        if (composition) {
            update_gbuffer_constants();
        }
    }
    int scale_index = static_cast<int>(resolution_scale);
    if (config.config_options("resolution scale", scale_index, RESOLUTION_SCALE_NAMES,
                              Properties::OptionsStyle::COMBO,
                              "per axis; an upscaler listening to the scene undoes it")) {
        resolution_scale = static_cast<uint32_t>(scale_index);
        needs_reconnect = true;
    }
    config.output_text("traced: {}x{}", extent.width, extent.height);

    config.st_separate("instance mask");
    for (uint32_t bit = 0; bit < 8; ++bit) {
        config.config_bool(std::to_string(bit), mask_enabled[bit]);
        if ((bit & 3u) != 3u)
            config.st_no_space();
    }

    config.st_separate();
    needs_reconnect |=
        config.config_enum("emission format", emission_format, Properties::OptionsStyle::COMBO);

    config.st_separate("Specular chain");
    needs_reconnect |=
        config.config_options("lobe", specular_lobe, {"sample", "reflect", "transmit", "fresnel"},
                              Properties::OptionsStyle::COMBO);
    needs_reconnect |= config.config_float("max roughness", specular_max_roughness,
                                           "surfaces above it close the chain", 0.01f, 0.f, 1.f);
    needs_reconnect |= config.config_int("max bounces", specular_max_bounces,
                                         "entering and leaving a closed surface takes two", 1, 8);
    needs_reconnect |= config.config_bool(
        "primary surface replacement", specular_psr,
        "the guides describe the surface the chain ends on instead of the one the camera sees");

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
