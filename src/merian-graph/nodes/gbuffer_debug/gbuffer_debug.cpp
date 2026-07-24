#include "merian-graph/nodes/gbuffer_debug/gbuffer_debug.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

namespace merian {

GBufferDebugNode::GBufferDebugNode() {}

DeviceSupportInfo GBufferDebugNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    // Scene parameter contains an AccelerationStructure; Slang emits RayTracingKHR for the AS
    // declaration when no ray-query op is reachable from this entry point. Enable
    // rayTracingPipeline opportunistically so the resulting SPIR-V loads.
    return DeviceSupportInfo::check(query_info, {}, {"rayTracingPipeline"});
}

void GBufferDebugNode::initialize(const ContextHandle& context,
                                  const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();
}

std::vector<InputConnectorDescriptor> GBufferDebugNode::describe_inputs() {
    return {{"scene", con_scene}, {"gbuffer", con_gbuffer, ConnectorAccess::compute_read}};
}

std::vector<OutputConnectorDescriptor>
GBufferDebugNode::describe_outputs(const NodeIOLayout& io_layout) {
    extent = io_layout[con_gbuffer]->get_create_info().extent;
    con_output = ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, extent);
    return {{"image", con_output, ConnectorAccess::compute_write}};
}

GBufferDebugNode::NodeStatusFlags
GBufferDebugNode::on_connected(const NodeIOLayout& io_layout,
                               [[maybe_unused]] const NodeIO& io,
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

    return {};
}

[[nodiscard]] GBufferDebugNode::NodeStatusFlags
GBufferDebugNode::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const auto& cmd = submission.get_cmd();
    const auto& scene = io[con_scene];
    const auto gbuf = io[con_gbuffer];
    if (!scene || !scene->is_ready())
        return {};

    if (!composition) {
        composition = SlangComposition::create();
        composition->add_composition(scene->get_composition());
        composition->add_module_from_path("merian-graph/nodes/gbuffer_debug/gbuffer_debug.slang",
                                          true);

        program = SlangProgram::create(compile_context, composition);
        entry_point = SlangProgramEntryPoint::create(program, "main");

        pipeline = Versioned<Pipeline>([this] {
            const auto ep = entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        pipeline.depends_on(entry_point);

        params = Versioned<ShaderObject>([this] {
            return entry_point->create_shader_object_for_parameter(context, "params",
                                                                   resource_allocator);
        });
        params.depends_on(entry_point);
    }

    const ShaderObjectAllocatorHandle& obj_allocator = info.get_shader_object_allocator();

    const auto ep = entry_point.get();
    const auto pipe = pipeline.get();
    const auto params_obj = params.get();

    auto cursor = params_obj->get_cursor();
    cursor["gbuffer"] = gbuf.r();
    cursor["output"] = io[con_output].get_texture();

    cmd->bind(pipe);
    ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
    ep->bind("params", params_obj, cmd, pipe, obj_allocator);
    cmd->push_constant(pipe, static_cast<int>(selected_field));

    cmd->dispatch(extent, 16, 16);
    return {};
}

GBufferDebugNode::NodeStatusFlags GBufferDebugNode::properties(Properties& config) {
    bool needs_reconnect = false;

    const std::vector<std::string> field_names = {
        // Shaded preview
        "Simple Shading",
        "Simple Shading with Albedo",
        "Albedo",
        // Normals & tangent frame
        "Normal",
        "Normal Texture",
        "Face Normal",
        "Tangent",
        "Bitangent",
        "Tangent W",
        // Material
        "Alpha",
        "Emissive",
        "Metallic",
        "Roughness",
        // IDs & barycentrics
        "Instance ID",
        "Geometry Index",
        "Geometry ID",
        "Primitive ID",
        "Material ID",
        "Barycentrics",
        // Depth & motion
        "Linear Z",
        "Grad Z",
        "Delta Z",
        "Motion Vectors",
        // Flags
        "Flat Shading Flag",
    };

    config.config_options("field", selected_field, field_names);

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
