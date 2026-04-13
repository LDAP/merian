#include "merian-nodes/nodes/gbuffer_rt/gbuffer.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

namespace merian {

GBufferRTNode::GBufferRTNode() {}

DeviceSupportInfo
GBufferRTNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return DeviceSupportInfo::check(
        query_info, {"accelerationStructure", "computeDerivativeGroupQuads"});
}

void GBufferRTNode::initialize(const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = ShaderCompileContext::create(context);
}

std::vector<InputConnectorDescriptor> GBufferRTNode::describe_inputs() {
    return {{"scene", con_scene}};
}

std::vector<OutputConnectorDescriptor>
GBufferRTNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    con_gbuffer = PtrOut<GBufferResource>::create();
    con_denoiser =
        ManagedVkImageOut::compute_read_write(vk::Format::eR32G32B32A32Uint, extent);
    con_hit_info =
        ManagedVkImageOut::compute_read_write(vk::Format::eR32G32B32A32Uint, extent);
    con_mv = ManagedVkImageOut::compute_read_write(vk::Format::eR16G16Sfloat, extent);

    return {
        {"gbuffer", con_gbuffer},
        {"denoiser", con_denoiser},
        {"hit_info", con_hit_info},
        {"mv", con_mv},
    };
}

GBufferRTNode::NodeStatusFlags GBufferRTNode::on_connected(
    [[maybe_unused]] const NodeIOLayout& io_layout,
    [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {

    pipeline = nullptr;
    program = nullptr;
    entry_point = nullptr;
    gbuffer_obj = nullptr;
    obj_allocator = nullptr;
    gbuffer_composition = nullptr;

    return {};
}

void GBufferRTNode::process(GraphRun& run,
                            [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                            const NodeIO& io) {
    const auto& cmd = run.get_cmd();
    const auto& scene = io[con_scene];
    if (!scene || !scene->has_geometry())
        return;

    // Lazily build program/pipeline from scene's composition
    if (!pipeline) {
        gbuffer_composition = SlangComposition::create();
        gbuffer_composition->add_module_from_path("merian-shaders/gbuffer.slang");

        auto composition = SlangComposition::create();
        composition->add_composition(scene->get_composition());
        composition->add_composition(gbuffer_composition);
        composition->add_module_from_path("merian-nodes/nodes/gbuffer_rt/gbuffer.slang",
                                          true /* with entry points */);

        program = SlangProgram::create(compile_context, composition);
        entry_point = SlangProgramEntryPoint::create(program, "main");

        auto pipe_layout = entry_point->get_pipeline_layout(context);
        auto vulkan_entry_point = entry_point->specialize();
        pipeline = ComputePipeline::create(pipe_layout, vulkan_entry_point);

        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, run.get_iterations_in_flight());
    }

    // Create shader object for GBufferImages
    if (!gbuffer_obj) {
        gbuffer_obj = entry_point->create_shader_object(context, "gbuffer", obj_allocator);
    }

    // Bind graph-managed textures to the shader object
    auto cursor = gbuffer_obj->get_cursor();
    cursor["tex0"] = io[con_denoiser].get_texture()->get_view();
    cursor["tex1"] = io[con_hit_info].get_texture()->get_view();
    cursor["tex2"] = io[con_mv].get_texture()->get_view();

    // Bind and dispatch
    obj_allocator->set_iteration(run.get_in_flight_index());
    cmd->bind(pipeline);
    entry_point->bind_entry_point_parameter("scene", scene->get_shader_object(), cmd, pipeline);
    entry_point->bind_entry_point_parameter("gbuffer", gbuffer_obj, cmd, pipeline);

    cmd->dispatch(extent, 16, 16);

    // Publish GBuffer resource for PtrIn consumers
    auto gbuf_resource = std::make_shared<GBufferResource>(
        io[con_denoiser].get_texture(),
        io[con_hit_info].get_texture(),
        io[con_mv].get_texture(),
        gbuffer_obj,
        gbuffer_composition);
    io[con_gbuffer] = gbuf_resource;
}

GBufferRTNode::NodeStatusFlags GBufferRTNode::properties(Properties& config) {
    bool needs_reconnect = false;
    needs_reconnect |= config.config_uint("width", &extent.width);
    needs_reconnect |= config.config_uint("height", &extent.height);

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
