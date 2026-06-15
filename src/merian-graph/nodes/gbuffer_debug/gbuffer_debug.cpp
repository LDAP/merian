#include "merian-graph/nodes/gbuffer_debug/gbuffer_debug.hpp"

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
    return {{"scene", con_scene}, {"gbuffer", con_gbuffer}};
}

std::vector<OutputConnectorDescriptor>
GBufferDebugNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    con_output = ManagedVkImageOut::compute_write(vk::Format::eR8G8B8A8Unorm, extent);
    return {{"image", con_output}};
}

GBufferDebugNode::NodeStatusFlags GBufferDebugNode::on_connected(
    [[maybe_unused]] const NodeIOLayout& io_layout,
    [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {

    // force the program graph to be rewired next process()
    composition = nullptr;
    obj_allocator = nullptr;

    return {};
}

void GBufferDebugNode::process(GraphRun& run,
                               [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                               const NodeIO& io) {
    const auto& cmd = run.get_cmd();
    const auto& scene = io[con_scene];
    const auto& gbuf = io[con_gbuffer];
    if (!scene || !gbuf || !scene->is_ready())
        return;

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
            return entry_point.get()->create_shader_object_for_parameter(context, "params",
                                                                         resource_allocator);
        });
        params.depends_on(entry_point);

        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, run.get_iterations_in_flight());
    }

    obj_allocator->set_iteration(run.get_in_flight_index());

    const auto ep = entry_point.get();
    const auto pipe = pipeline.get();
    const auto params_obj = params.get();

    auto cursor = params_obj->get_cursor();
    cursor["gbuffer"] = gbuf->get_shader_object();
    cursor["output"] = io[con_output].get_texture();

    cmd->bind(pipe);
    ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
    ep->bind("params", params_obj, cmd, pipe, obj_allocator);
    cmd->push_constant(pipe, static_cast<int>(selected_field));

    cmd->dispatch(extent, 16, 16);
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

    needs_reconnect |= config.config_uint("width", &extent.width);
    needs_reconnect |= config.config_uint("height", &extent.height);

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
