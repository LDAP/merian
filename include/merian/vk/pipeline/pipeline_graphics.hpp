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
                     const RenderPassHandle& renderpass,
                     const uint32_t subpass,
                     const vk::PipelineCreateFlags flags = {},
                     const std::shared_ptr<Pipeline>& base_pipeline = {})
        : Pipeline(pipeline_layout->get_context(), pipeline_layout, flags),
          base_pipeline(base_pipeline) {

        SPDLOG_DEBUG("create GraphicsPipeline ({})", fmt::ptr(this));
        const vk::GraphicsPipelineCreateInfo info{flags,
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
                                                  *renderpass,
                                                  subpass,
                                                  base_pipeline ? base_pipeline->get_pipeline()
                                                                : nullptr,
                                                  0

        };

        // Hm. This is a bug in the API there should not be .value
        pipeline = context->device.createGraphicsPipeline(context->pipeline_cache, info).value;
    }

    ~GraphicsPipeline() {
        SPDLOG_DEBUG("destroy GraphicsPipeline ({})", fmt::ptr(this));
        context->device.destroyPipeline(pipeline);
    }

    // Overrides
    // ---------------------------------------------------------------------------

    virtual vk::PipelineBindPoint get_pipeline_bind_point() const override {
        return vk::PipelineBindPoint::eGraphics;
    }

    // ---------------------------------------------------------------------------

  private:
    const std::shared_ptr<Pipeline> base_pipeline;
};
using GraphicsPipelineHandle = std::shared_ptr<GraphicsPipeline>;

} // namespace merian
