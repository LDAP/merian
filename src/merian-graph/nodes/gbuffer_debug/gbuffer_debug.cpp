#include "merian-graph/nodes/gbuffer_debug/gbuffer_debug.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

namespace merian {

namespace {

struct DebugField {
    const char* name;
    std::vector<GBufferField> fields;
};

// indexed by the shader's selected_field
const std::vector<DebugField> FIELDS = {
    // Shaded preview
    {"Simple Shading", {GBufferField::Normal}},
    {"Simple Shading with Albedo", {GBufferField::Hit}},
    {"Albedo", {GBufferField::Albedo}},
    // Normals & tangent frame
    {"Normal", {GBufferField::Normal}},
    {"Normal Texture", {GBufferField::Hit, GBufferField::Normal}},
    {"Face Normal", {GBufferField::Hit}},
    {"Tangent", {GBufferField::Hit}},
    {"Bitangent", {GBufferField::Hit}},
    {"Tangent W", {GBufferField::Hit}},
    // Material
    {"Alpha", {GBufferField::Hit}},
    {"Emissive", {GBufferField::Hit}},
    {"Metallic", {GBufferField::Hit}},
    {"Roughness", {GBufferField::Hit}},
    // IDs & barycentrics
    {"Instance ID", {GBufferField::Hit}},
    {"Geometry Index", {GBufferField::Hit}},
    {"Geometry ID", {GBufferField::Hit}},
    {"Primitive ID", {GBufferField::Hit}},
    {"Material ID", {GBufferField::Hit}},
    {"Barycentrics", {GBufferField::Hit}},
    // Depth & motion
    {"Linear Z", {GBufferField::LinearZ}},
    {"Grad Z", {GBufferField::GradZ}},
    {"Delta Z", {GBufferField::DeltaZ}},
    {"Motion Vectors", {GBufferField::MotionVectors}},
    // Flags
    {"Flat Shading Flag", {GBufferField::Hit}},
    // Lobes & depths
    {"Diffuse Albedo", {GBufferField::DiffuseAlbedo}},
    {"Specular Albedo", {GBufferField::SpecularAlbedo}},
    {"Specular Roughness", {GBufferField::Roughness}},
    {"View Depth", {GBufferField::ViewDepth}},
    {"Projected Depth", {GBufferField::ProjectedDepth}},
};

} // namespace

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
    con_gbuffer = GBufferIn::create({{FIELDS[selected_field].fields}});
    return {{"scene", con_scene}, {"gbuffer", con_gbuffer, ConnectorAccess::compute_read}};
}

std::vector<OutputConnectorDescriptor>
GBufferDebugNode::describe_outputs(const NodeIOLayout& io_layout) {
    extent = io_layout[con_gbuffer]->get_create_info().extent;
    con_output = ManagedVkImageOut::create(vk::Format::eR16G16B16A16Sfloat, extent);
    return {{"image", con_output, ConnectorAccess::compute_write}};
}

GBufferDebugNode::NodeStatusFlags
GBufferDebugNode::on_connected(const NodeIOLayout& io_layout,
                               const NodeIO& io,
                               [[maybe_unused]] const NodeConnectionInfo& info,
                               [[maybe_unused]] Submission& submission) {

    // force the program graph to be rewired next process()
    composition = nullptr;
    gbuffer_composition = io[con_gbuffer]->get_layout()->get_composition();

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
        composition->add_composition(gbuffer_composition);
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
    std::vector<std::string> field_names;
    field_names.reserve(FIELDS.size());
    for (const DebugField& field : FIELDS) {
        field_names.emplace_back(field.name);
    }

    if (config.config_options("field", selected_field, field_names)) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
