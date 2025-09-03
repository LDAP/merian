#pragma once

#include "merian/vk/shader/entry_point.hpp"
#include "pipeline_graphics.hpp"

#include "merian/vk/shader/shader_module.hpp"

namespace merian {

/*
 * Builder for Graphics pipelines with sensible defaults.
 *
 * - No vertrex bindings and attributes
 * - Triangle list topology
 * - No primitive restart
 * - 0 tessellation patch control points
 * - no viewport
 * - fill polygons
 * - cull backfacing
 * - counter clockwise winding order
 * - depth bias disabled
 * - line width 1.0
 * - no multisampling
 * - no sample shading
 * - sample mask of NULL (means all bits set)
 * - alphaToCoverage | alphaToOne is false
 * - depth test and write disabled, vk::CompareOp::eLess operation
 * - stencil test disabled, prepared with vk::StencilOp::eIncrementAndClamp for both sides
 * - logic op disable but prepared with vk::LogicOp::eClear
 *
 * - no dynamic state
 */
class GraphicsPipelineBuilder {
  public:
    // --- Vertex Input State ---

    GraphicsPipelineBuilder&
    vertex_input_flags(const vk::PipelineVertexInputStateCreateFlags vertex_input_flags);

    GraphicsPipelineBuilder&
    vertex_input_add_binding(const vk::VertexInputBindingDescription& vertex_binding_descriptions);

    GraphicsPipelineBuilder& vertex_input_add_attribute(
        const vk::VertexInputAttributeDescription& vertex_attribute_descriptions);

    // --- Vertex Assembly State ---

    GraphicsPipelineBuilder&
    input_assembly_flags(const vk::PipelineInputAssemblyStateCreateFlags flags);

    GraphicsPipelineBuilder& input_assembly_topology(
        const vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList);

    GraphicsPipelineBuilder& input_assembly_primitive_restart(const bool enable = true);

    // --- Tessellation State ---

    GraphicsPipelineBuilder&
    tessellation_flags(const vk::PipelineTessellationStateCreateFlags flags);

    GraphicsPipelineBuilder& tessellation_patch_control_points(const uint32_t patch_control_points);

    // --- Viewport ---

    GraphicsPipelineBuilder&
    viewport_flags(const vk::PipelineViewportStateCreateFlags viewport_create_flags);

    // sets the scissor by default to match the viewport.
    GraphicsPipelineBuilder&
    viewport_add(const float width,
                 const float height,
                 const float x = 0,
                 const float y = 0,
                 const float min_depth = 0.f,
                 const float max_depth = 1.f,
                 const std::optional<vk::Offset2D>& scissor_offset = std::nullopt,
                 const std::optional<vk::Extent2D>& scissor_extent = std::nullopt);

    GraphicsPipelineBuilder& viewport_add(const vk::Viewport& viewport, const vk::Rect2D& scissor);

    GraphicsPipelineBuilder& viewport_add(const vk::Extent3D& extent,
                                          const float min_depth = 0.f,
                                          const float max_depth = 1.f);

    // --- Rasterizer ---

    GraphicsPipelineBuilder&
    rasterizer_flags(const vk::PipelineRasterizationStateCreateFlags flags);

    GraphicsPipelineBuilder& rasterizer_depth_clamp(const bool enable = true);

    GraphicsPipelineBuilder& rasterizer_discard(const bool enable = true);

    GraphicsPipelineBuilder&
    rasterizer_polygon_mode(const vk::PolygonMode mode = vk::PolygonMode::eFill);

    GraphicsPipelineBuilder&
    rasterizer_cull_mode(const vk::CullModeFlags mode = vk::CullModeFlagBits::eBack);

    GraphicsPipelineBuilder&
    rasterizer_front_face(const vk::FrontFace front_face = vk::FrontFace::eCounterClockwise);

    // automatically sets depth_bias to enable
    GraphicsPipelineBuilder& rasterizer_depth_bias(const float depth_bias_constant_factor = 0,
                                                   const float depth_bias_clamp = 0,
                                                   const float depth_bias_slope_factor = 0);

    GraphicsPipelineBuilder& rasterizer_depth_bias_enable(const bool enable = true);

    // --- Multisample ---

    GraphicsPipelineBuilder& multisample_flags(const vk::PipelineMultisampleStateCreateFlags flags);

    GraphicsPipelineBuilder&
    multisample_count(const vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1);

    // automatically enables sample shading
    // sample shading: Normally only the depth test is multisampled but the fragment shader is still
    // only executed once (MSAA). With sample shading the fragment shader is executed multiple times
    // (SSAA). This means sample shading comes at a high cost. This setting sets the minimum
    // fraction of sample shading.
    GraphicsPipelineBuilder& multisample_sample_shading(const float min_sample_shading = 1.0);

    GraphicsPipelineBuilder& multisample_shading_enable(const bool enable = true);

    GraphicsPipelineBuilder& multisample_sample_alpha_to_coverage(const bool enable = true);

    GraphicsPipelineBuilder& multisample_sample_alpha_to_one(const bool enable = true);

    // --- Depth & Stencil Test ---

    GraphicsPipelineBuilder&
    depth_stencil_flags(const vk::PipelineDepthStencilStateCreateFlags flags);

    GraphicsPipelineBuilder& depth_test_enable(const bool enable = true);

    GraphicsPipelineBuilder& depth_write_enable(const bool enable = true);

    // calling will automatically enable depth tests (but not write!!)
    GraphicsPipelineBuilder& depth_compare(const vk::CompareOp compare_op = vk::CompareOp::eLess);

    GraphicsPipelineBuilder& stencil_test_enable(const bool enable = true);

    // calling will automatically enable stencil tests
    GraphicsPipelineBuilder&
    stencil_operation(const vk::StencilOp front_face = vk::StencilOp::eIncrementAndClamp,
                      const vk::StencilOp back_face = vk::StencilOp::eIncrementAndClamp);

    GraphicsPipelineBuilder& depth_bounds_test_enable(const bool enable = true);

    // calling will automatically enable depth bounds test
    GraphicsPipelineBuilder& depth_bounds(const float min = 0., const float max = 1.);

    // --- Color Blend Test ---

    GraphicsPipelineBuilder& blend_flags(const vk::PipelineColorBlendStateCreateFlags flags);

    GraphicsPipelineBuilder& blend_logic_op_enable(const bool enable = true);

    // calling will automatically enable logic ops
    // requires the logic op feature
    GraphicsPipelineBuilder& blend_logic_op(const vk::LogicOp logic_op);

    // all must be identical if the independentBlend feature is not enabled.
    GraphicsPipelineBuilder&
    blend_add_attachment(const vk::PipelineColorBlendAttachmentState& state);

    GraphicsPipelineBuilder& blend_add_attachment(
        const vk::Bool32 blend_enable = VK_TRUE,
        const vk::BlendFactor src_color_blend_factor = vk::BlendFactor::eOne,
        const vk::BlendFactor dst_color_blend_factor = vk::BlendFactor::eOne,
        const vk::BlendOp color_blend_op = vk::BlendOp::eAdd,
        const vk::BlendFactor src_alpha_blend_factor = vk::BlendFactor::eOne,
        const vk::BlendFactor dst_alpha_blend_factor = vk::BlendFactor::eOne,
        const vk::BlendOp alpha_blend_op = vk::BlendOp::eAdd,
        const vk::ColorComponentFlags color_write_mask = vk::ColorComponentFlagBits::eR |
                                                         vk::ColorComponentFlagBits::eG |
                                                         vk::ColorComponentFlagBits::eB |
                                                         vk::ColorComponentFlagBits::eA);

    // --- Dynamic States ---

    GraphicsPipelineBuilder& dyanmic_state_add(const vk::DynamicState state);

    // --- Shader Modules ---

    GraphicsPipelineBuilder& set_vertex_shader(const VulkanEntryPointHandle& vertex_shader);

    GraphicsPipelineBuilder&
    set_geometry_shader(const VulkanEntryPointHandle& geometry_shader);

    GraphicsPipelineBuilder& set_mesh_shader(const VulkanEntryPointHandle& mesh_shader);

    GraphicsPipelineBuilder&
    set_fragment_shader(const VulkanEntryPointHandle& fragment_shader);

    GraphicsPipelineBuilder&
    set_tessellation_shader(const VulkanEntryPointHandle& tessellation_control_shader,
                            const VulkanEntryPointHandle& tessellation_evaluation_shader);

    // --- Build ---

    GraphicsPipelineHandle build(const PipelineLayoutHandle& pipeline_layout,
                                 const RenderPassHandle& renderpass,
                                 const uint32_t subpass = 0,
                                 const std::shared_ptr<Pipeline>& opt_base_pipeline = {});

  private:
    // Shaders
    std::optional<VulkanEntryPointHandle> vertex_shader = std::nullopt;
    std::optional<VulkanEntryPointHandle> geometry_shader = std::nullopt;
    std::optional<VulkanEntryPointHandle> mesh_shader = std::nullopt;
    std::optional<VulkanEntryPointHandle> fragment_shader = std::nullopt;
    std::optional<VulkanEntryPointHandle> tessellation_control_shader = std::nullopt;
    std::optional<VulkanEntryPointHandle> tessellation_evaluation_shader = std::nullopt;

    // Vertex Input State
    vk::PipelineVertexInputStateCreateFlags vertex_input_create_flags;
    std::vector<vk::VertexInputBindingDescription> vertex_input_bindings;
    std::vector<vk::VertexInputAttributeDescription> vertex_input_attributes;

    vk::PipelineInputAssemblyStateCreateInfo input_assembly_state{
        {}, vk::PrimitiveTopology::eTriangleList, VK_FALSE};
    vk::PipelineTessellationStateCreateInfo tessellation_state{{}, 0};

    vk::PipelineViewportStateCreateFlags viewport_create_flags;
    std::vector<vk::Viewport> viewports;
    std::vector<vk::Rect2D> scissors;

    vk::PipelineRasterizationStateCreateInfo rasterization_state{{},
                                                                 VK_FALSE,
                                                                 VK_FALSE,
                                                                 vk::PolygonMode::eFill,
                                                                 vk::CullModeFlagBits::eBack,
                                                                 vk::FrontFace::eCounterClockwise,
                                                                 VK_FALSE,
                                                                 0,
                                                                 0,
                                                                 0,
                                                                 1.0};
    vk::PipelineMultisampleStateCreateInfo multisample_state{
        {}, vk::SampleCountFlagBits::e1, VK_FALSE, 0, nullptr, VK_FALSE, VK_FALSE};
    vk::PipelineDepthStencilStateCreateInfo depth_stencil_state{{},
                                                                VK_FALSE,
                                                                VK_FALSE,
                                                                vk::CompareOp::eLess,
                                                                VK_FALSE,
                                                                VK_FALSE,
                                                                vk::StencilOp::eIncrementAndClamp,
                                                                vk::StencilOp::eIncrementAndClamp,
                                                                0.f,
                                                                0.f};
    vk::PipelineColorBlendStateCreateInfo color_blend_state{{}, VK_FALSE, vk::LogicOp::eClear};
    std::vector<vk::PipelineColorBlendAttachmentState> attachment_blend_states;

    vk::PipelineDynamicStateCreateFlags dynamic_state_create_flags;
    std::vector<vk::DynamicState> dynamic_states;

    vk::PipelineCreateFlags flags = {};
    std::shared_ptr<Pipeline> base_pipeline = {};
};
} // namespace merian
