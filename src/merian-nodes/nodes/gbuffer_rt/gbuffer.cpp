#include "merian-nodes/nodes/gbuffer_rt/gbuffer.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

namespace merian {

GBufferRTNode::GBufferRTNode() {}

DeviceSupportInfo GBufferRTNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return DeviceSupportInfo::check(query_info,
                                    {"accelerationStructure", "computeDerivativeGroupQuads"});
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
    con_gbuffer = PtrOut<GBuffer>::create();
    con_denoiser = ManagedVkImageOut::compute_read_write(vk::Format::eR32G32B32A32Uint, extent);
    con_hit_info = ManagedVkImageOut::compute_read_write(vk::Format::eR32G32B32A32Uint, extent);
    con_mv = ManagedVkImageOut::compute_read_write(vk::Format::eR16G16Sfloat, extent);
    con_albedo = ManagedVkImageOut::compute_read_write(vk::Format::eR32G32B32A32Sfloat, extent);
    con_emission = ManagedVkImageOut::compute_read_write(vk::Format::eR32G32B32A32Sfloat, extent);

    return {
        {"gbuffer", con_gbuffer}, {"denoiser", con_denoiser}, {"hit_info", con_hit_info},
        {"mv", con_mv},           {"albedo", con_albedo},    {"emission", con_emission},
    };
}

GBufferRTNode::NodeStatusFlags GBufferRTNode::on_connected(
    [[maybe_unused]] const NodeIOLayout& io_layout,
    [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {

    pipeline = nullptr;
    program = nullptr;
    entry_point = nullptr;
    gbuffer_obj = nullptr;
    globals_obj = nullptr;
    obj_allocator = nullptr;

    return {};
}

void GBufferRTNode::process(GraphRun& run,
                            [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                            const NodeIO& io) {
    const auto& cmd = run.get_cmd();
    const auto& scene = io[con_scene];

    // Lazily build program/pipeline from scene's composition
    if (!program) {
        auto composition = SlangComposition::create();
        composition->add_composition(scene->get_composition());
        composition->add_module_from_path("merian-nodes/nodes/gbuffer_rt/gbuffer.slang", true);

        program = SlangProgram::create(compile_context, composition);
        entry_point = SlangProgramEntryPoint::create(program, "main");

        // entry_point rebuilds when the scene's composition (or any of its
        // sub-compositions) changes. The cached pipeline + globals shader
        // object embed the old reflection, so drop them on every rebuild.
        entry_point->on_changed(entry_point, [this] {
            pipeline = nullptr;
            globals_obj = nullptr;
        });

        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, run.get_iterations_in_flight());
    }

    if (!pipeline) {
        auto pipe_layout = entry_point->get_pipeline_layout(context);
        auto vulkan_entry_point = entry_point->specialize();
        pipeline = ComputePipeline::create(pipe_layout, vulkan_entry_point);
    }

    obj_allocator->set_iteration(run.get_in_flight_index());

    // Create shader object for GBufferImages
    if (!gbuffer_obj) {
        gbuffer_obj = std::make_shared<GBuffer>(compile_context, context, obj_allocator, extent);
    }

    if (!globals_obj) {
        globals_obj = entry_point->create_global_shader_object(context, obj_allocator);
    }

    // Bind graph-managed textures to the shader object
    gbuffer_obj->set_resources(
        io[con_denoiser].get_texture()->get_view(), io[con_hit_info].get_texture()->get_view(),
        io[con_mv].get_texture()->get_view(), io[con_albedo].get_texture()->get_view());

    globals_obj->get_cursor()["emission"] = io[con_emission].get_texture();

    if (scene->has_geometry()) {
        cmd->bind(pipeline);
        entry_point->bind_entry_point_parameter("scene", scene->get_shader_object(), cmd, pipeline);
        entry_point->bind_entry_point_parameter("gbuffer", gbuffer_obj->get_write_shader_object(),
                                                cmd, pipeline);
        entry_point->bind_global_parameter(globals_obj, cmd, pipeline);
        cmd->dispatch(extent, 16, 16);
    }

    // cmd->barrier({io[con_denoiser]->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal),
    //               io[con_hit_info]->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal),
    //               io[con_mv]->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal),
    //               io[con_albedo]->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal)});

    io[con_gbuffer] = gbuffer_obj;
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
