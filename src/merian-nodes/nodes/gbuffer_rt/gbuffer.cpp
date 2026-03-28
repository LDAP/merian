#include "merian-nodes/nodes/gbuffer_rt/gbuffer.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"

namespace merian {

GBufferRTNode::GBufferRTNode() {}

void GBufferRTNode::initialize(const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;

    compile_context = ShaderCompileContext::create(context);
    program = SlangProgram::create(compile_context, "merian-nodes/nodes/gbuffer_rt/gbuffer.slang");
    entry_point = SlangProgramEntryPoint::create(program, "main");
}

std::vector<InputConnectorDescriptor> GBufferRTNode::describe_inputs() {
    return {{"src", con_src}};
}

std::vector<OutputConnectorDescriptor>
GBufferRTNode::describe_outputs(const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_src]->get_create_info_or_throw();
    const vk::Extent3D extent = create_info.extent;

    con_out = ManagedVkImageOut::compute_read_write(vk::Format::eR32G32B32A32Uint, extent);

    return {{"gbuffer", con_out}};
}

GBufferRTNode::NodeStatusFlags GBufferRTNode::on_connected(
    [[maybe_unused]] const NodeIOLayout& io_layout,
    [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {

    auto pipe_layout = entry_point->get_pipeline_layout(context);

    auto vulkan_entry_point = entry_point->specialize();
    pipeline = ComputePipeline::create(pipe_layout, vulkan_entry_point);

    // Reset so process() re-creates with correct iterations_in_flight
    obj_allocator = nullptr;
    params = nullptr;
    manual_cb_obj = nullptr;
    manual_pb_obj = nullptr;
    replace_cb_obj = nullptr;
    frame_count = 0;

    return {};
}

void GBufferRTNode::process(GraphRun& run,
                            [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                            const NodeIO& io) {
    const auto& cmd = run.get_cmd();

    // Lazily create the allocator with the correct iterations_in_flight
    if (!obj_allocator) {
        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, run.get_iterations_in_flight());
    }

    // =====================================================================
    // First frame: create ShaderObject and set all static values once
    // =====================================================================
    if (!params) {
        params = entry_point->create_shader_object(context, "params", obj_allocator);

        auto cursor = params->get_cursor();

        // --- Static values (set once, must persist across frames) ---

        // Value: MaterialParams — metallic is static, roughness updated per-frame
        cursor["material"]["metallic"] = 1.0f;

        // CB: LightInfo — direction is static, intensity updated per-frame
        float direction[3] = {0.f, 1.f, 0.f};
        cursor["light"]["direction"].write(direction, sizeof(direction));

        // Nested PB: NestedParams — weight is static
        cursor["nested"]["weight"] = 0.3f;

        // CB in nested PB: NestedLightInfo — all static
        float color[3] = {1.f, 0.f, 0.f};
        cursor["nested"]["nested_light"]["color"].write(color, sizeof(color));
        cursor["nested"]["nested_light"]["power"] = 100.0f;

        // Deep PB: DeepParams — deep_value updated per-frame
        // CB in deep PB: DeepLightInfo — all static
        float ambient[3] = {0.2f, 0.2f, 0.2f};
        cursor["nested"]["deep"]["deep_light"]["ambient"].write(ambient, sizeof(ambient));
        cursor["nested"]["deep"]["deep_light"]["strength"] = 7.0f;

        // Deepest PB: DeeperParams — static
        cursor["nested"]["deep"]["deeper"]["deepest_value"] = 99.0f;

        // CB-in-CB test — static
        cursor["cb_in_cb"]["outer_val"] = 5.0f;
        cursor["cb_in_cb"]["inner"]["inner_val"] = 3.0f;

        // ===========================================================
        // TEST: Explicit ShaderObject creation + assignment (CB)
        // Create a ShaderObject manually, set its value, install via set_sub_object.
        // ===========================================================
        manual_cb_obj = params->create_sub_object("manual_cb");
        manual_cb_obj->get_cursor()["manual_cb_val"] = 11.0f;
        params->set_sub_object("manual_cb", manual_cb_obj);

        // ===========================================================
        // TEST: Explicit ShaderObject creation + assignment (PB)
        // manual_pb_tex is set per-frame (needs graph image).
        // ===========================================================
        manual_pb_obj = params->create_sub_object("manual_pb");
        manual_pb_obj->get_cursor()["manual_pb_val"] = 22.0f;
        params->set_sub_object("manual_pb", manual_pb_obj);

        // ===========================================================
        // TEST: Reassignment — initial CB, will be replaced later
        // ===========================================================
        replace_cb_obj = params->create_sub_object("replace_cb");
        replace_cb_obj->get_cursor()["replace_val"] = 33.0f;
        params->set_sub_object("replace_cb", replace_cb_obj);
    }

    // =====================================================================
    // Per-frame updates (incremental: only changed values and descriptors)
    // =====================================================================

    auto cursor = params->get_cursor();

    // TEST: Incremental ordinary data update (value binding, set 0)
    cursor["material"]["roughness"] = 0.8f;

    // TEST: Incremental CB value update (CB staging re-upload)
    cursor["light"]["intensity"] = 1.5f;

    // TEST: Incremental nested PB value update (deep in the tree)
    cursor["nested"]["deep"]["deep_value"] = 42.0f;

    // Resources from the graph change each frame
    cursor["input"] = io[con_src].get_texture()->get_view();
    cursor["output"] = io[con_out].get_texture()->get_view();
    cursor["nested"]["extra_map"] = io[con_src].get_texture()->get_view();
    cursor["tex_only"]["tex"] = io[con_src].get_texture()->get_view();

    // TEST: Resource in manually-assigned PB (must work after explicit assignment)
    manual_pb_obj->get_cursor()["manual_pb_tex"] = io[con_src].get_texture()->get_view();

    // ===========================================================
    // TEST: Sub-object reassignment after 3 frames
    // Replace replace_cb with a new object containing a different value.
    // Tests that the old sub-object is cleanly replaced and the new
    // one's buffer descriptor + staging data are used correctly.
    // ===========================================================
    frame_count++;
    if (frame_count == 3) {
        replace_cb_obj = params->create_sub_object("replace_cb");
        replace_cb_obj->get_cursor()["replace_val"] = 77.0f; // changed from 33.0
        params->set_sub_object("replace_cb", replace_cb_obj);
    }

    // Bind pipeline and descriptor sets, then dispatch
    obj_allocator->set_iteration(run.get_in_flight_index());
    cmd->bind(pipeline);
    entry_point->bind("params", params, obj_allocator, cmd, pipeline);

    const vk::Extent3D extent = io[con_out]->get_extent();
    cmd->dispatch(extent, 16, 16);
}

} // namespace merian
