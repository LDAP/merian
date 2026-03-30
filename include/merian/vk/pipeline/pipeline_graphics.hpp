#pragma once

#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/renderpass/renderpass.hpp"

namespace merian {

class GraphicsPipeline : public Pipeline {

  public:
    GraphicsPipeline(const std::vector<vk::PipelineShaderStageCreateInfo>& stages,
                     const vk::PipelineVertexInputStateCreateInfo& pVertexInputState,
                     const vk::PipelineInputAssemblyStateCreateInfo& pInputAssemblyState,
                     const vk::PipelineTessellationStateCreateInfo& pTessellationState,
                     const vk::PipelineViewportStateCreateInfo& pViewportState,
                     const vk::PipelineRasterizationStateCreateInfo& pRasterizationState,
                     const vk::PipelineMultisampleStateCreateInfo& pMultisampleState,
                     const vk::PipelineDepthStencilStateCreateInfo& pDepthStencilState,
                     const vk::PipelineColorBlendStateCreateInfo& pColorBlendState,
                     const vk::PipelineDynamicStateCreateInfo& pDynamicState,
                     const PipelineLayoutHandle& pipeline_layout,
                     const RenderPassHandle& renderpass = {},
                     const uint32_t subpass = 0,
                     const vk::PipelineCreateFlags flags = {},
                     const std::shared_ptr<Pipeline>& base_pipeline = {},
                     const void* pNext = nullptr)
        : Pipeline(pipeline_layout->get_context(), pipeline_layout, flags),
          base_pipeline(base_pipeline),
          active_shader_stages(compute_stage_flags(stages)) {

        SPDLOG_DEBUG("create GraphicsPipeline ({})", fmt::ptr(this));
        vk::GraphicsPipelineCreateInfo info{flags,
                                            stages,
                                            &pVertexInputState,
                                            &pInputAssemblyState,
                                            &pTessellationState,
                                            &pViewportState,
                                            &pRasterizationState,
                                            &pMultisampleState,
                                            &pDepthStencilState,
                                            &pColorBlendState,
                                            &pDynamicState,
                                            *pipeline_layout,
                                            renderpass ? *renderpass : vk::RenderPass{VK_NULL_HANDLE},
                                            subpass,
                                            base_pipeline ? base_pipeline->get_pipeline()
                                                          : nullptr,
                                            0

        };
        info.pNext = pNext;

        // Hm. This is a bug in the API there should not be .value
        pipeline = context->get_device()
                       ->get_device()
                       .createGraphicsPipeline(context->get_device()->get_pipeline_cache(), info)
                       .value;
    }

    ~GraphicsPipeline() {
        SPDLOG_DEBUG("destroy GraphicsPipeline ({})", fmt::ptr(this));
        context->get_device()->get_device().destroyPipeline(pipeline);
    }

    // Overrides
    // ---------------------------------------------------------------------------

    vk::PipelineBindPoint get_pipeline_bind_point() const override {
        return vk::PipelineBindPoint::eGraphics;
    }

    vk::PipelineStageFlags get_pipeline_stage_flags() const override {
        return active_shader_stages;
    }

    vk::PipelineStageFlags2 get_pipeline_stage_flags2() const override {
        return static_cast<vk::PipelineStageFlags2>(
            static_cast<VkPipelineStageFlags>(active_shader_stages));
    }

    // ---------------------------------------------------------------------------

  private:
    static vk::PipelineStageFlags
    compute_stage_flags(const std::vector<vk::PipelineShaderStageCreateInfo>& stages) {
        vk::PipelineStageFlags result;
        for (const auto& stage : stages) {
            switch (stage.stage) {
            case vk::ShaderStageFlagBits::eVertex:
                result |= vk::PipelineStageFlagBits::eVertexShader;
                break;
            case vk::ShaderStageFlagBits::eFragment:
                result |= vk::PipelineStageFlagBits::eFragmentShader;
                break;
            case vk::ShaderStageFlagBits::eGeometry:
                result |= vk::PipelineStageFlagBits::eGeometryShader;
                break;
            case vk::ShaderStageFlagBits::eTessellationControl:
                result |= vk::PipelineStageFlagBits::eTessellationControlShader;
                break;
            case vk::ShaderStageFlagBits::eTessellationEvaluation:
                result |= vk::PipelineStageFlagBits::eTessellationEvaluationShader;
                break;
            case vk::ShaderStageFlagBits::eMeshEXT:
                result |= vk::PipelineStageFlagBits::eMeshShaderEXT;
                break;
            case vk::ShaderStageFlagBits::eTaskEXT:
                result |= vk::PipelineStageFlagBits::eTaskShaderEXT;
                break;
            default:
                break;
            }
        }
        return result;
    }

    const std::shared_ptr<Pipeline> base_pipeline;
    const vk::PipelineStageFlags active_shader_stages;
};
using GraphicsPipelineHandle = std::shared_ptr<GraphicsPipeline>;

} // namespace merian
