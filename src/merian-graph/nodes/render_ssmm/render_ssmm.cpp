#include "merian-graph/nodes/render_ssmm/render_ssmm.hpp"

#include "merian-graph/nodes/render_ssmm/render_ssmm.slangh"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"

#include <fmt/format.h>

#include <random>

namespace merian {

namespace {
constexpr const char* SHADER_MODULE = "merian-graph/nodes/render_ssmm/render_ssmm.slang";
} // namespace

RenderSSMM::RenderSSMM() = default;

DeviceSupportInfo RenderSSMM::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_module_from_path(SHADER_MODULE, true);
    const auto program = SlangProgram::create(query_info.compile_context, composition);
    return DeviceSupportInfo::check(query_info, {"rayTracingPipeline"}, {"rayQuery"}) &
           program.get()->query_device_support(query_info);
}

void RenderSSMM::initialize(const ContextHandle& context,
                            const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();
}

vk::BufferCreateInfo RenderSSMM::ssmc_buffer_create_info() const {
    return vk::BufferCreateInfo{{},
                                vk::DeviceSize(extent.width) * extent.height * sizeof(SSMCState),
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                    vk::BufferUsageFlagBits::eTransferDst};
}

void RenderSSMM::update_render_constants() {
    composition->add_module_from_string(
        "render_ssmm_constants",
        fmt::format("namespace merian {{\n"
                    "export static const int merian_ssmm_spp = {};\n"
                    "export static const float merian_ssmm_bsdf_p = {:f};\n"
                    "export static const float merian_ssmm_ml_prior_n = {:f};\n"
                    "export static const uint merian_ssmm_ml_max_n = {}u;\n"
                    "export static const float merian_ssmm_ml_min_alpha = {:f};\n"
                    "export static const int merian_ssmm_smis_group_size = {};\n"
                    "}}",
                    spp, surf_bsdf_p, ml_prior_n, ml_max_n, ml_min_alpha, smis_group_size));
}

std::vector<InputConnectorDescriptor> RenderSSMM::describe_inputs() {
    return {{"scene", con_scene},
            {"gbuffer", con_gbuffer, ConnectorAccess::ray_tracing_read},
            {"prev_ssmc", con_prev_ssmc,
             ConnectorAccess::ray_tracing_read | ConnectorAccess::transfer_dst, 1}};
}

std::vector<OutputConnectorDescriptor>
RenderSSMM::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    con_irradiance = ManagedVkImageOut::create(vk::Format::eR32G32B32A32Sfloat, extent);
    con_ssmc = ManagedVkBufferOut::create(ssmc_buffer_create_info());
    return {{"irradiance", con_irradiance, ConnectorAccess::ray_tracing_write},
            {"ssmc", con_ssmc,
             ConnectorAccess::ray_tracing_read_write | ConnectorAccess::transfer_dst}};
}

RenderSSMM::NodeStatusFlags
RenderSSMM::on_connected(const NodeIOLayout& io_layout,
                         [[maybe_unused]] const NodeIO& io,
                         [[maybe_unused]] const NodeConnectionInfo& info,
                         [[maybe_unused]] Submission& submission) {
    composition = nullptr;
    obj_allocator = nullptr;
    ssmc_needs_reset = true;

    if (randomize_seed) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<uint32_t> dist;
        seed = dist(rng);
    }

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

    return {};
}

[[nodiscard]] RenderSSMM::NodeStatusFlags
RenderSSMM::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const auto& cmd = submission.get_cmd();
    const auto& scene = io[con_scene];
    const auto gbuf = io[con_gbuffer];
    if (!scene || !scene->is_ready())
        return {};

    if (!composition) {
        composition = SlangComposition::create();
        composition->add_composition(scene->get_composition());
        composition->add_module_from_path(SHADER_MODULE, true);
        update_render_constants();

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

        params = Versioned<ShaderObject>([this] {
            return entry_point->create_shader_object_for_parameter(context, "params",
                                                                   resource_allocator);
        });
        params.depends_on(entry_point);

        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, info.get_iterations_in_flight());
    }

    obj_allocator->set_iteration(info.get_in_flight_index());

    // RESET MARKOV CHAINS (also clear previous frame)
    if (ssmc_needs_reset) {
        ssmc_needs_reset = false;
        cmd->fill(io[con_ssmc]);
        cmd->fill(io[con_prev_ssmc]);

        const std::array<vk::BufferMemoryBarrier, 2> barriers = {
            io[con_ssmc]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                         vk::AccessFlagBits::eShaderRead),
            io[con_prev_ssmc]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                              vk::AccessFlagBits::eShaderRead),
        };

        cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eRayTracingShaderKHR, barriers);
    }

    const auto ep = entry_point.get();
    const auto pipe = pipeline.get();
    const auto params_obj = params.get();

    auto cursor = params_obj->get_cursor();
    cursor["gbuffer"] = gbuf.r();
    cursor["irradiance"] = io[con_irradiance].get_texture();

    SSMMPushConstant pc{};
    pc.ssmc_prev = io[con_prev_ssmc]->get_device_address();
    pc.ssmc_out = io[con_ssmc]->get_device_address();
    pc.frame = static_cast<uint32_t>(info.get_iteration());
    pc.seed = seed;

    cmd->bind(pipe);
    ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
    ep->bind("params", params_obj, cmd, pipe, obj_allocator);
    cmd->push_constant(pipe, pc);

    cmd->trace_rays(sbt.get(), extent);
    return {};
}

RenderSSMM::NodeStatusFlags RenderSSMM::properties(Properties& config) {
    bool needs_reconnect = false;
    bool constants_changed = false;

    config.st_separate("General");
    config.config_bool("randomize seed", randomize_seed, "randomize seed at every graph build");
    if (!randomize_seed) {
        config.config_uint("seed", seed, "");
    } else {
        config.output_text(fmt::format("seed: {}", seed));
    }

    config.st_separate();
    constants_changed |= config.config_int("spp", spp, "samples per pixel", 0, 15);
    constants_changed |=
        config.config_percent("BSDF Prob", surf_bsdf_p, "the probability to use BSDF sampling");

    config.st_separate("MLE estimation");
    constants_changed |= config.config_float("prior N", ml_prior_n, "", 0.01);
    constants_changed |= config.config_uint("max N", ml_max_n, "");
    constants_changed |= config.config_float("min alpha", ml_min_alpha, "", 0.01);

    config.st_separate("SMIS");
    constants_changed |= config.config_uint("group size", smis_group_size, "");

    config.st_separate("Resolution");
    needs_reconnect |= config.config_uint("width", &extent.width);
    needs_reconnect |= config.config_uint("height", &extent.height);

    if (constants_changed && composition) {
        update_render_constants();
    }

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
