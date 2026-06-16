#include "merian-graph/nodes/render_restir_di/render_restir_di.hpp"

#include "merian-graph/nodes/render_restir_di/render_restir_di.slangh"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"

namespace merian {

namespace {
constexpr std::array<const char*, RenderRestirDI::PassCount> PASS_MODULES = {
    "merian-graph/nodes/render_restir_di/generate.slang",
    "merian-graph/nodes/render_restir_di/temporal.slang",
    "merian-graph/nodes/render_restir_di/spatial.slang",
    "merian-graph/nodes/render_restir_di/shade.slang"};
} // namespace

RenderRestirDI::RenderRestirDI() = default;

DeviceSupportInfo RenderRestirDI::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return DeviceSupportInfo::check(query_info, {"rayTracingPipeline"}, {"rayQuery"});
}

void RenderRestirDI::initialize(const ContextHandle& context,
                                const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();
}

vk::BufferCreateInfo RenderRestirDI::reservoir_buffer_create_info() const {
    return vk::BufferCreateInfo{{},
                                vk::DeviceSize(extent.width) * extent.height * sizeof(RestirReservoir),
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                    vk::BufferUsageFlagBits::eTransferDst};
}

std::vector<InputConnectorDescriptor> RenderRestirDI::describe_inputs() {
    return {{"scene", con_scene},
            {"gbuffer", con_gbuffer},
            {"prev_gbuffer", con_prev_gbuffer},
            {"prev_reservoirs", con_prev_reservoirs}};
}

std::vector<OutputConnectorDescriptor>
RenderRestirDI::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    con_irradiance = ManagedVkImageOut::compute_write(vk::Format::eR32G32B32A32Sfloat, extent);
    con_reservoirs = ManagedVkBufferOut::compute_write(reservoir_buffer_create_info(), true);
    return {{"irradiance", con_irradiance}, {"reservoirs", con_reservoirs}};
}

RenderRestirDI::NodeStatusFlags
RenderRestirDI::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                             [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {
    // force the program graph to be rewired next process()
    compositions = {};
    obj_allocator = nullptr;

    pong_buffer = resource_allocator->create_buffer(reservoir_buffer_create_info(),
                                                    MemoryMappingType::NONE, "ReSTIR DI reservoirs");

    return {};
}

void RenderRestirDI::process(GraphRun& run,
                             [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                             const NodeIO& io) {
    const auto& cmd = run.get_cmd();
    const auto& scene = io[con_scene];
    const auto& gbuf = io[con_gbuffer];
    const auto& prev_gbuf = io[con_prev_gbuffer];
    if (!scene || !gbuf || !scene->is_ready())
        return;

    if (!compositions[Generate]) {
        for (uint32_t p = 0; p < PassCount; p++) {
            compositions[p] = SlangComposition::create();
            compositions[p]->add_composition(scene->get_composition());
            compositions[p]->add_module_from_path(PASS_MODULES[p], true);

            programs[p] = SlangProgram::create(compile_context, compositions[p]);
            entry_points[p] = SlangProgramEntryPoint::create(programs[p], "main");

            pipelines[p] = Versioned<RayTracingPipeline>([this, p] {
                const auto ep = entry_points[p].get();
                return RayTracingPipelineBuilder()
                    .add_raygen_group(ep->specialize())
                    .build(ep->get_pipeline_layout(context));
            });
            pipelines[p].depends_on(entry_points[p]);

            sbts[p] = Versioned<ShaderBindingTable>([this, p] {
                return ShaderBindingTable::create(pipelines[p].get(), resource_allocator);
            });
            sbts[p].depends_on(pipelines[p]);

            params[p] = Versioned<ShaderObject>([this, p] {
                return entry_points[p].get()->create_shader_object_for_parameter(
                    context, "params", resource_allocator);
            });
            params[p].depends_on(entry_points[p]);
        }

        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, run.get_iterations_in_flight());
    }

    obj_allocator->set_iteration(run.get_in_flight_index());

    const bool spatial_enabled = spatial_iterations > 0;
    const vk::DeviceAddress scratch = pong_buffer->get_device_address();
    const vk::DeviceAddress out = io[con_reservoirs]->get_device_address();
    const vk::DeviceAddress prev = io[con_prev_reservoirs]->get_device_address();
    const vk::DeviceAddress gen_buffer = spatial_enabled ? scratch : out;
    const BufferHandle gen_handle = spatial_enabled ? pong_buffer : BufferHandle(io[con_reservoirs]);
    const BufferHandle out_handle = io[con_reservoirs];

    RestirPushConstant pc{};
    pc.reservoirs_prev = prev;
    pc.frame = static_cast<uint32_t>(run.get_iteration());
    pc.seed = seed;
    pc.spp = spp;
    pc.temporal_bias_correction = temporal_bias_correction;
    pc.spatial_bias_correction = spatial_bias_correction;
    pc.temporal_clamp_m = temporal_clamp_m;
    pc.spatial_radius = spatial_radius;
    pc.spatial_iterations = spatial_iterations;
    pc.temporal_normal_reject_cos = temporal_normal_reject_cos;
    pc.temporal_depth_reject = temporal_depth_reject;
    pc.spatial_normal_reject_cos = spatial_normal_reject_cos;
    pc.spatial_depth_reject = spatial_depth_reject;
    pc.boiling_filter_strength = boiling_filter_strength;
    pc.flags = (apply_mv ? RestirFlagApplyMotion : 0u) |
               (visibility_shade ? RestirFlagVisibilityShade : 0u) |
               (temporal_enable ? RestirFlagTemporalEnable : 0u);

    const auto bind_params = [&](const Pass p) {
        const auto obj = params[p].get();
        auto cursor = obj->get_cursor();
        cursor["gbuffer"] = gbuf->get_shader_object();
        cursor["prev_gbuffer"] = (prev_gbuf ? prev_gbuf : gbuf)->get_shader_object();
        cursor["irradiance"] = io[con_irradiance].get_texture();
        return obj;
    };

    const auto run_pass = [&](const Pass p, const vk::DeviceAddress in_addr,
                              const vk::DeviceAddress out_addr) {
        const auto pipe = pipelines[p].get();
        pc.pass = static_cast<uint32_t>(p);
        pc.reservoirs_in = in_addr;
        pc.reservoirs_out = out_addr;
        cmd->bind(pipe);
        entry_points[p].get()->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
        entry_points[p].get()->bind("params", bind_params(p), cmd, pipe, obj_allocator);
        cmd->push_constant(pipe, pc);
        cmd->trace_rays(sbts[p].get(), extent);
    };

    const auto sync = [&](const BufferHandle& buffer) {
        const auto bar = buffer->buffer_barrier(
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
        cmd->barrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                     vk::PipelineStageFlagBits::eRayTracingShaderKHR, {bar});
    };

    run_pass(Generate, 0, gen_buffer);
    sync(gen_handle);

    if (temporal_enable && run.get_iteration() > 0) {
        run_pass(Temporal, gen_buffer, gen_buffer);
        sync(gen_handle);
    }

    if (spatial_enabled) {
        run_pass(Spatial, scratch, out);
        sync(out_handle);
    }

    run_pass(Shade, out, out);
}

RenderRestirDI::NodeStatusFlags RenderRestirDI::properties(Properties& config) {
    bool needs_reconnect = false;

    config.st_separate("Generate");
    config.config_int("samples per pixel", spp, "BSDF-sampled candidates per pixel.", 0, 32);
    config.config_uint("seed", seed, "Base seed for the per-pixel RNG.");

    config.st_separate("Temporal reuse");
    config.config_bool("enable temporal reuse", temporal_enable);
    float temporal_angle = std::acos(temporal_normal_reject_cos);
    config.config_angle("normal threshold", temporal_angle,
                        "Reject reprojections with normals farther apart.", 0, 180);
    temporal_normal_reject_cos = std::cos(temporal_angle);
    config.config_percent("depth threshold", temporal_depth_reject,
                          "Reject reprojections with depths farther apart (relative to the max).");
    config.config_int("clamp M", temporal_clamp_m,
                      "Clamp the temporal history length. 0 disables.");
    config.config_options("bias correction", temporal_bias_correction,
                          {"none", "basic", "raytraced"});
    config.config_bool("apply motion", apply_mv,
                       "Extrapolate the light position from its velocity. Reduces flicker on "
                       "moving lights but biases motion.");
    config.config_percent("boiling filter", boiling_filter_strength,
                          "Discard outlier reservoirs within a subgroup. 0 disables.");

    config.st_separate("Spatial reuse");
    config.config_int("iterations", spatial_iterations, "Neighbors resampled per pixel.", 0, 8);
    float spatial_angle = std::acos(spatial_normal_reject_cos);
    config.config_angle("normal threshold", spatial_angle,
                        "Reject neighbors with normals farther apart.", 0, 180);
    spatial_normal_reject_cos = std::cos(spatial_angle);
    config.config_percent("depth threshold", spatial_depth_reject,
                          "Reject neighbors with depths farther apart (relative to the max).");
    config.config_int("radius", spatial_radius, "Pixel radius for neighbor sampling.", 0, 100);
    config.config_options("bias correction", spatial_bias_correction,
                          {"none", "basic", "raytraced"});

    config.st_separate("Shade");
    config.config_bool("visibility", visibility_shade, "Trace a shadow ray before shading.");

    config.st_separate("Resolution");
    needs_reconnect |= config.config_uint("width", &extent.width);
    needs_reconnect |= config.config_uint("height", &extent.height);

    if (needs_reconnect)
        return NEEDS_RECONNECT;
    return {};
}

} // namespace merian
