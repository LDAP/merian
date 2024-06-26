#pragma once

#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class ComputePipeline : public Pipeline {

  public:
    ComputePipeline(
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const std::shared_ptr<ShaderModule>& shader_module,
        const SpecializationInfoHandle specialization_info = MERIAN_SPECIALIZATION_INFO_NONE,
        const char* shader_module_entry_point = "main",
        const vk::PipelineShaderStageCreateFlags stage_flags = {},
        const vk::PipelineCreateFlags flags = {},
        const std::shared_ptr<Pipeline>& shared_base_pipeline_handle = {},
        const int32_t base_pipeline_index = {})
        : Pipeline(pipeline_layout->get_context(), pipeline_layout), shader_module(shader_module),
          shared_base_pipeline_handle(shared_base_pipeline_handle) {
        SPDLOG_DEBUG("create ComputePipeline ({})", fmt::ptr(this));

        vk::PipelineShaderStageCreateInfo stage = shader_module->get_shader_stage_create_info(
            vk::ShaderStageFlagBits::eCompute, specialization_info, shader_module_entry_point,
            stage_flags);
        const vk::Pipeline base_pipeline_handle =
            shared_base_pipeline_handle ? shared_base_pipeline_handle->get_pipeline() : nullptr;
        vk::ComputePipelineCreateInfo info{flags, stage, *pipeline_layout, base_pipeline_handle,
                                           base_pipeline_index};
        // Hm. This is a bug in the API there should not be .value
        pipeline = context->device.createComputePipeline(context->pipeline_cache, info).value;
    }

    ~ComputePipeline() {
        SPDLOG_DEBUG("destroy ComputePipeline ({})", fmt::ptr(this));
        context->device.destroyPipeline(pipeline);
    }

    // Overrides
    // ---------------------------------------------------------------------------

    virtual vk::PipelineBindPoint get_pipeline_bind_point() const {
        return vk::PipelineBindPoint::eCompute;
    }

    // ---------------------------------------------------------------------------

    const std::shared_ptr<ShaderModule>& get_module() const {
        return shader_module;
    }

  private:
    const std::shared_ptr<ShaderModule> shader_module;
    const std::shared_ptr<Pipeline> shared_base_pipeline_handle;
};

} // namespace merian
