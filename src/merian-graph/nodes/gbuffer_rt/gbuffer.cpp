#include "merian-graph/nodes/gbuffer_rt/gbuffer.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"

#include <fmt/format.h>

namespace merian {

GBufferRTNode::GBufferRTNode() {}

DeviceSupportInfo GBufferRTNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
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
    con_gbuffer = ShaderObjectOut<GBufferObject>::create(
        [extent = extent] { return std::make_shared<GBufferObject>(extent); });
    con_emission = ManagedVkImageOut::create(vk::Format::eR32G32B32A32Sfloat, extent);

    return {
        {"gbuffer", con_gbuffer, ConnectorAccess::ray_tracing_write},
        {"emission", con_emission, ConnectorAccess::ray_tracing_write},
    };
}

GBufferRTNode::NodeStatusFlags GBufferRTNode::on_connected(const NodeConnectedInfo& info) {
    const NodeIOLayout& io_layout = info.io_layout;

    // force the program graph to be rewired next process()
    composition = nullptr;
    obj_allocator = nullptr;

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

void GBufferRTNode::process(GraphRun& run, const NodeIO& io) {
    const auto& cmd = run.get_cmd();
    const auto& scene = io[con_scene];

    emission_connected = io.is_connected(con_emission);

    if (!composition) {
        composition = SlangComposition::create();
        composition->add_composition(scene->get_composition());
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

        globals_obj = Versioned<ShaderObject>([this] {
            return entry_point->create_global_shader_object(context, resource_allocator);
        });
        globals_obj.depends_on(entry_point);

        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, run.get_iterations_in_flight());
    }

    obj_allocator->set_iteration(run.get_in_flight_index());

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
}

void GBufferRTNode::update_gbuffer_constants() {
    composition->add_module_from_string("gbuffer_constants",
                                        fmt::format("namespace merian {{ export static const bool "
                                                    "merian_gbuffer_write_emission = {}; }}",
                                                    emission_connected ? "true" : "false"));
}

GBufferRTNode::NodeStatusFlags GBufferRTNode::properties(Properties& config) {
    bool needs_reconnect = false;
    needs_reconnect |= config.config_uint("width", &extent.width);
    needs_reconnect |= config.config_uint("height", &extent.height);

    config.st_separate("instance mask");
    for (uint32_t bit = 0; bit < 8; ++bit) {
        config.config_bool(std::to_string(bit), mask_enabled[bit]);
        if ((bit & 3u) != 3u)
            config.st_no_space();
    }

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
