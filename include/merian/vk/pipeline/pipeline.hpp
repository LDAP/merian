#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"

#include "vulkan/vulkan.hpp"
#include <spdlog/spdlog.h>

namespace merian {

class Pipeline {

  public:
    Pipeline(const SharedContext& context, const std::shared_ptr<PipelineLayout>& pipeline_layout)
        : context(context), pipeline_layout(pipeline_layout) {}

    virtual ~Pipeline() {};

    // ---------------------------------------------------------------------------

    operator const vk::Pipeline() const {
        return pipeline;
    }

    const vk::Pipeline& get_pipeline() {
        return pipeline;
    }

    const std::shared_ptr<PipelineLayout>& get_layout() {
        return pipeline_layout;
    }

    // ---------------------------------------------------------------------------

    virtual void bind(vk::CommandBuffer& cmd) = 0;

    virtual void bind_descriptor_set(vk::CommandBuffer& cmd,
                                     std::shared_ptr<DescriptorSet>& descriptor_set) = 0;

  protected:
    const SharedContext context;
    const std::shared_ptr<PipelineLayout> pipeline_layout;
    vk::Pipeline pipeline;
};

} // namespace merian
