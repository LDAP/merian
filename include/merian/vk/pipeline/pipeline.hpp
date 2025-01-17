#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"

#include <spdlog/spdlog.h>

namespace merian {

class Pipeline : public std::enable_shared_from_this<Pipeline>, public Object  {

  public:
    Pipeline(const ContextHandle& context, const std::shared_ptr<PipelineLayout>& pipeline_layout)
        : context(context), pipeline_layout(pipeline_layout) {}

    virtual ~Pipeline(){};

    // ---------------------------------------------------------------------------

    operator const vk::Pipeline&() const {
        return pipeline;
    }

    const vk::Pipeline& get_pipeline() {
        return pipeline;
    }

    const std::shared_ptr<PipelineLayout>& get_layout() {
        return pipeline_layout;
    }

    // ---------------------------------------------------------------------------

    virtual vk::PipelineBindPoint get_pipeline_bind_point() const = 0;

  protected:
    const ContextHandle context;
    const std::shared_ptr<PipelineLayout> pipeline_layout;
    vk::Pipeline pipeline;
};

using PipelineHandle = std::shared_ptr<Pipeline>;

} // namespace merian
