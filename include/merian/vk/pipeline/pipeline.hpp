#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"

#include <spdlog/spdlog.h>

namespace merian {

class Pipeline : public std::enable_shared_from_this<Pipeline>, public Object {

  public:
    Pipeline(const ContextHandle& context,
             const std::shared_ptr<PipelineLayout>& pipeline_layout,
             const vk::PipelineCreateFlags flags)
        : context(context), pipeline_layout(pipeline_layout), flags(flags) {}

    virtual ~Pipeline() {};

    // ---------------------------------------------------------------------------

    operator const vk::Pipeline&() const {
        return pipeline;
    }

    const vk::Pipeline& get_pipeline() const {
        return pipeline;
    }

    const std::shared_ptr<PipelineLayout>& get_layout() const {
        return pipeline_layout;
    }

    bool supports_descriptor_buffer() const {
        return bool(flags & vk::PipelineCreateFlagBits::eDescriptorBufferEXT);
    }

    bool supports_descriptor_set() const {
        // https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCreateFlagBits.html#
        return !supports_descriptor_buffer();
    }

    // ---------------------------------------------------------------------------

    virtual vk::PipelineBindPoint get_pipeline_bind_point() const = 0;

  protected:
    const ContextHandle context;
    const PipelineLayoutHandle pipeline_layout;
    const vk::PipelineCreateFlags flags;

    vk::Pipeline pipeline;
};

using PipelineHandle = std::shared_ptr<Pipeline>;

} // namespace merian
