#pragma once

#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/shader/entry_point.hpp"

namespace merian {

class ComputePipeline : public Pipeline {
  private:
    ComputePipeline(const PipelineLayoutHandle& pipeline_layout,
                    const VulkanEntryPointHandle& entry_point,
                    const vk::PipelineCreateFlags flags = {},
                    const PipelineHandle& base_pipeline = {},
                    const void* pNext = nullptr)
        : Pipeline(pipeline_layout->get_context(), pipeline_layout), entry_point(entry_point),
          base_pipeline(base_pipeline) {

        assert(entry_point->get_stage() == vk::ShaderStageFlagBits::eCompute);

        SPDLOG_DEBUG("create ComputePipeline ({})", fmt::ptr(this));

        const vk::ComputePipelineCreateInfo info{
            flags,
            entry_point->get_shader_stage_create_info(context),
            *pipeline_layout,
            base_pipeline ? base_pipeline->get_pipeline() : nullptr,
            0,
            pNext,
        };
        // Hm. This is a bug in the API there should not be .value
        pipeline = context->device.createComputePipeline(context->pipeline_cache, info).value;
    }

  public:
    ~ComputePipeline() {
        SPDLOG_DEBUG("destroy ComputePipeline ({})", fmt::ptr(this));
        context->device.destroyPipeline(pipeline);
    }

    // ---------------------------------------------------------------------------

    static PipelineHandle create(const PipelineLayoutHandle& pipeline_layout,
                                 const VulkanEntryPointHandle& entry_point,
                                 const vk::PipelineCreateFlags flags = {},
                                 const PipelineHandle& base_pipeline = {},
                                 const void* pNext = nullptr) {
        return PipelineHandle(
            new ComputePipeline(pipeline_layout, entry_point, flags, base_pipeline, pNext));
    }

    // shortcurt for create(..., entry_point.specialize(spec_info),...)
    static PipelineHandle create(const PipelineLayoutHandle& pipeline_layout,
                                 const EntryPointHandle& unspecialized_entry_point,
                                 const SpecializationInfoHandle& specialization_info,
                                 const vk::PipelineCreateFlags flags = {},
                                 const PipelineHandle& base_pipeline = {},
                                 const void* pNext = nullptr) {
        return PipelineHandle(new ComputePipeline(
            pipeline_layout, unspecialized_entry_point->specialize(specialization_info), flags,
            base_pipeline, pNext));
    }

    // Overrides
    // ---------------------------------------------------------------------------

    virtual vk::PipelineBindPoint get_pipeline_bind_point() const override {
        return vk::PipelineBindPoint::eCompute;
    }

    // ---------------------------------------------------------------------------

    const EntryPointHandle& get_entry_point() const {
        return entry_point;
    }

  private:
    const EntryPointHandle entry_point;
    const std::shared_ptr<Pipeline> base_pipeline;
};

} // namespace merian
