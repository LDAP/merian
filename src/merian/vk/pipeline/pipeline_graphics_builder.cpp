#include "merian/vk/pipeline/pipeline_graphics_builder.hpp"

namespace merian {

// --- Vertex Input State ---

GraphicsPipelineBuilder& GraphicsPipelineBuilder::vertex_input_flags(
    const vk::PipelineVertexInputStateCreateFlags vertex_input_flags) {
    vertex_input_create_flags = vertex_input_flags;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::vertex_input_add_binding(
    const vk::VertexInputBindingDescription& vertex_binding_descriptions) {
    vertex_input_bindings.emplace_back(vertex_binding_descriptions);
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::vertex_input_add_attribute(
    const vk::VertexInputAttributeDescription& vertex_attribute_descriptions) {
    vertex_input_attributes.emplace_back(vertex_attribute_descriptions);
    return *this;
}

// --- Vertex Assembly State ---

GraphicsPipelineBuilder& GraphicsPipelineBuilder::input_assembly_flags(
    const vk::PipelineInputAssemblyStateCreateFlags flags) {
    input_assembly_state.flags = flags;
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::input_assembly_topology(const vk::PrimitiveTopology topology) {
    input_assembly_state.topology = topology;
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::input_assembly_primitive_restart(const bool enable) {
    input_assembly_state.primitiveRestartEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

// --- Tessellation State ---

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::tessellation_flags(const vk::PipelineTessellationStateCreateFlags flags) {
    tessellation_state.flags = flags;
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::tessellation_patch_control_points(const uint32_t patch_control_points) {
    tessellation_state.patchControlPoints = patch_control_points;
    return *this;
}

// --- Viewport ---

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::viewport_flags(const vk::PipelineViewportStateCreateFlags flags) {
    viewport_create_flags = flags;
    return *this;
}

// sets the scissor by default to match the viewport.
GraphicsPipelineBuilder&
GraphicsPipelineBuilder::viewport_add(const float width,
                                      const float height,
                                      const float x,
                                      const float y,
                                      const float min_depth,
                                      const float max_depth,
                                      const std::optional<vk::Offset2D>& scissor_offset,
                                      const std::optional<vk::Extent2D>& scissor_extent) {
    viewports.emplace_back(vk::Viewport{x, y, width, height, min_depth, max_depth});
    scissors.emplace_back(vk::Rect2D{
        scissor_offset.value_or(vk::Offset2D{static_cast<int32_t>(x), static_cast<int32_t>(y)}),
        scissor_extent.value_or(
            vk::Extent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)})});

    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::viewport_add(const vk::Viewport& viewport,
                                                               const vk::Rect2D& scissor) {
    viewports.emplace_back(viewport);
    scissors.emplace_back(scissor);
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::viewport_add(const vk::Extent3D& extent,
                                                               const float min_depth,
                                                               const float max_depth) {
    viewports.emplace_back(vk::Viewport{0, 0, static_cast<float>(extent.width),
                                        static_cast<float>(extent.height), min_depth, max_depth});
    scissors.emplace_back(vk::Rect2D{{}, {extent.width, extent.height}});
    return *this;
}

// --- Rasterizer ---

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::rasterizer_flags(const vk::PipelineRasterizationStateCreateFlags flags) {
    rasterization_state.flags = flags;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::rasterizer_depth_clamp(const bool enable) {
    rasterization_state.depthClampEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::rasterizer_discard(const bool enable) {
    rasterization_state.rasterizerDiscardEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::rasterizer_polygon_mode(const vk::PolygonMode mode) {
    rasterization_state.polygonMode = mode;
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::rasterizer_cull_mode(const vk::CullModeFlags mode) {
    rasterization_state.cullMode = mode;
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::rasterizer_front_face(const vk::FrontFace front_face) {
    rasterization_state.frontFace = front_face;
    return *this;
}

// automatically sets depth_bias to enable
GraphicsPipelineBuilder&
GraphicsPipelineBuilder::rasterizer_depth_bias(const float depth_bias_constant_factor,
                                               const float depth_bias_clamp,
                                               const float depth_bias_slope_factor) {
    rasterization_state.depthBiasConstantFactor = depth_bias_constant_factor;
    rasterization_state.depthBiasClamp = depth_bias_clamp;
    rasterization_state.depthBiasSlopeFactor = depth_bias_slope_factor;
    return rasterizer_depth_bias_enable(true);
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::rasterizer_depth_bias_enable(const bool enable) {
    rasterization_state.depthBiasEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

// --- Multisample ---

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::multisample_flags(const vk::PipelineMultisampleStateCreateFlags flags) {
    multisample_state.flags = flags;
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::multisample_count(const vk::SampleCountFlagBits samples) {
    multisample_state.rasterizationSamples = samples;
    return *this;
}

// automatically enables sample shading
// sample shading: Normally only the depth test is multisampled but the fragment shader is still
// only executed once (MSAA). With sample shading the fragment shader is executed multiple times
// (SSAA). This means sample shading comes at a high cost. This setting sets the minimum
// fraction of sample shading.
GraphicsPipelineBuilder&
GraphicsPipelineBuilder::multisample_sample_shading(const float min_sample_shading) {
    multisample_state.minSampleShading = min_sample_shading;
    return multisample_shading_enable(true);
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::multisample_shading_enable(const bool enable) {
    multisample_state.sampleShadingEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::multisample_sample_alpha_to_coverage(const bool enable) {
    multisample_state.alphaToCoverageEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::multisample_sample_alpha_to_one(const bool enable) {
    multisample_state.alphaToOneEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

// --- Depth & Stencil Test ---

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::depth_stencil_flags(const vk::PipelineDepthStencilStateCreateFlags flags) {
    depth_stencil_state.flags = flags;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::depth_test_enable(const bool enable) {
    depth_stencil_state.depthTestEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::depth_write_enable(const bool enable) {
    depth_stencil_state.depthWriteEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

// calling will automatically enable depth tests (but not write!!)
GraphicsPipelineBuilder& GraphicsPipelineBuilder::depth_compare(const vk::CompareOp compare_op) {
    depth_stencil_state.depthCompareOp = compare_op;
    return depth_test_enable(true);
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::stencil_test_enable(const bool enable) {
    depth_stencil_state.stencilTestEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

// calling will automatically enable stencil tests
GraphicsPipelineBuilder& GraphicsPipelineBuilder::stencil_operation(const vk::StencilOp front_face,
                                                                    const vk::StencilOp back_face) {
    depth_stencil_state.front = front_face;
    depth_stencil_state.back = back_face;
    return stencil_test_enable(true);
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::depth_bounds_test_enable(const bool enable) {
    depth_stencil_state.depthBoundsTestEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

// calling will automatically enable depth bounds test
GraphicsPipelineBuilder& GraphicsPipelineBuilder::depth_bounds(const float min, const float max) {
    depth_stencil_state.minDepthBounds = min;
    depth_stencil_state.maxDepthBounds = max;
    return depth_bounds_test_enable(true);
}

// --- Color Blend Test ---

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::blend_flags(const vk::PipelineColorBlendStateCreateFlags flags) {
    color_blend_state.flags = flags;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::blend_logic_op_enable(const bool enable) {
    color_blend_state.logicOpEnable = static_cast<vk::Bool32>(enable);
    return *this;
}

// calling will automatically enable logic ops
// requires the logic op feature
GraphicsPipelineBuilder& GraphicsPipelineBuilder::blend_logic_op(const vk::LogicOp logic_op) {
    color_blend_state.logicOp = logic_op;
    return blend_logic_op_enable(true);
}

// all must be identical if the independentBlend feature is not enabled.
GraphicsPipelineBuilder&
GraphicsPipelineBuilder::blend_add_attachment(const vk::PipelineColorBlendAttachmentState& state) {
    attachment_blend_states.emplace_back(state);
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::blend_add_attachment(const vk::Bool32 blend_enable,
                                              const vk::BlendFactor src_color_blend_factor,
                                              const vk::BlendFactor dst_color_blend_factor,
                                              const vk::BlendOp color_blend_op,
                                              const vk::BlendFactor src_alpha_blend_factor,
                                              const vk::BlendFactor dst_alpha_blend_factor,
                                              const vk::BlendOp alpha_blend_op,
                                              const vk::ColorComponentFlags color_write_mask) {
    return blend_add_attachment(vk::PipelineColorBlendAttachmentState{
        blend_enable,
        src_color_blend_factor,
        dst_color_blend_factor,
        color_blend_op,
        src_alpha_blend_factor,
        dst_alpha_blend_factor,
        alpha_blend_op,
        color_write_mask,
    });
}

// --- Dynamic States ---

GraphicsPipelineBuilder& GraphicsPipelineBuilder::dyanmic_state_add(const vk::DynamicState state) {
    dynamic_states.emplace_back(state);
    return *this;
}

// --- Shader Modules ---

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::set_vertex_shader(const VulkanEntryPointHandle& vertex_shader) {
    assert(vertex_shader->get_stage() == vk::ShaderStageFlagBits::eVertex);
    this->vertex_shader.emplace(vertex_shader);
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::set_geometry_shader(const VulkanEntryPointHandle& geometry_shader) {
    assert(geometry_shader->get_stage() == vk::ShaderStageFlagBits::eGeometry);
    this->geometry_shader.emplace(geometry_shader);
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::set_mesh_shader(const VulkanEntryPointHandle& mesh_shader) {
    assert(mesh_shader->get_stage() == vk::ShaderStageFlagBits::eMeshEXT);
    this->mesh_shader.emplace(mesh_shader);
    return *this;
}

GraphicsPipelineBuilder&
GraphicsPipelineBuilder::set_fragment_shader(const VulkanEntryPointHandle& fragment_shader) {
    assert(fragment_shader->get_stage() == vk::ShaderStageFlagBits::eFragment);
    this->fragment_shader.emplace(fragment_shader);
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::set_tessellation_shader(
    const VulkanEntryPointHandle& tessellation_control_shader,
    const VulkanEntryPointHandle& tessellation_evaluation_shader) {
    assert(tessellation_control_shader->get_stage() ==
           vk::ShaderStageFlagBits::eTessellationControl);
    assert(tessellation_evaluation_shader->get_stage() ==
           vk::ShaderStageFlagBits::eTessellationEvaluation);

    this->tessellation_control_shader.emplace(tessellation_control_shader);
    this->tessellation_evaluation_shader.emplace(tessellation_evaluation_shader);
    return *this;
}

// --- Build ---

GraphicsPipelineHandle
GraphicsPipelineBuilder::build(const PipelineLayoutHandle& pipeline_layout,
                               const RenderPassHandle& renderpass,
                               const uint32_t subpass,
                               const std::shared_ptr<Pipeline>& base_pipeline) {
    assert(pipeline_layout);
    assert(vertex_shader);
    assert(fragment_shader);
    assert(renderpass);

    const vk::PipelineVertexInputStateCreateInfo vertex_input_state{
        vertex_input_create_flags, vertex_input_bindings, vertex_input_attributes};

    const vk::PipelineViewportStateCreateInfo viewport_state{viewport_create_flags, viewports,
                                                             scissors};
    color_blend_state.pAttachments = attachment_blend_states.data();
    color_blend_state.attachmentCount = attachment_blend_states.size();
    const vk::PipelineDynamicStateCreateInfo dynamic_state{dynamic_state_create_flags,
                                                           dynamic_states};

    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    if (vertex_shader) {
        stages.emplace_back(
            vertex_shader.value()->get_shader_stage_create_info(pipeline_layout->get_context()));
    }
    if (geometry_shader) {
        stages.emplace_back(
            geometry_shader.value()->get_shader_stage_create_info(pipeline_layout->get_context()));
    }
    if (mesh_shader) {
        stages.emplace_back(
            mesh_shader.value()->get_shader_stage_create_info(pipeline_layout->get_context()));
    }
    if (fragment_shader) {
        stages.emplace_back(
            fragment_shader.value()->get_shader_stage_create_info(pipeline_layout->get_context()));
    }
    assert((tessellation_control_shader && tessellation_evaluation_shader) ||
           (!tessellation_control_shader && !tessellation_evaluation_shader));
    if (tessellation_control_shader) {
        stages.emplace_back(tessellation_control_shader.value()->get_shader_stage_create_info(
            pipeline_layout->get_context()));
        stages.emplace_back(tessellation_evaluation_shader.value()->get_shader_stage_create_info(
            pipeline_layout->get_context()));
    }

    return std::make_shared<GraphicsPipeline>(
        stages, vertex_input_state, input_assembly_state, tessellation_state, viewport_state,
        rasterization_state, multisample_state, depth_stencil_state, color_blend_state,
        dynamic_state, pipeline_layout, renderpass, subpass, flags, base_pipeline);
}

} // namespace merian
