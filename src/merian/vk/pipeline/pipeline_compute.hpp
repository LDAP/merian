#pragma once

#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class ComputePipeline : public Pipeline {

  public:
    ComputePipeline(
        const PipelineLayoutHandle& pipeline_layout,
        const ShaderModuleHandle& shader_module,
        const SpecializationInfoHandle specialization_info = MERIAN_SPECIALIZATION_INFO_NONE,
        const char* shader_module_entry_point = "main",
        const vk::PipelineShaderStageCreateFlags stage_flags = {},
        const vk::PipelineCreateFlags flags = {},
        const PipelineHandle& base_pipeline = {})
        : Pipeline(pipeline_layout->get_context(), pipeline_layout), shader_module(shader_module),
          base_pipeline(base_pipeline) {
        SPDLOG_DEBUG("create ComputePipeline ({})", fmt::ptr(this));

        const vk::PipelineShaderStageCreateInfo stage = shader_module->get_shader_stage_create_info(
            vk::ShaderStageFlagBits::eCompute, specialization_info, shader_module_entry_point,
            stage_flags);

        const vk::ComputePipelineCreateInfo info{
            flags, stage, *pipeline_layout, base_pipeline ? base_pipeline->get_pipeline() : nullptr,
            0,
        };
        // Hm. This is a bug in the API there should not be .value
        pipeline = context->device.createComputePipeline(context->pipeline_cache, info).value;
    }

    ~ComputePipeline() {
        SPDLOG_DEBUG("destroy ComputePipeline ({})", fmt::ptr(this));
        context->device.destroyPipeline(pipeline);
    }

    // Overrides
    // ---------------------------------------------------------------------------

    virtual vk::PipelineBindPoint get_pipeline_bind_point() const override {
        return vk::PipelineBindPoint::eCompute;
    }

    // ---------------------------------------------------------------------------

    const std::shared_ptr<ShaderModule>& get_module() const {
        return shader_module;
    }

  private:
    const std::shared_ptr<ShaderModule> shader_module;
    const std::shared_ptr<Pipeline> base_pipeline;
};

} // namespace merian
