#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"

#include <spdlog/spdlog.h>

namespace merian {

class Pipeline : public std::enable_shared_from_this<Pipeline> {

  public:
    Pipeline(const SharedContext& context, const std::shared_ptr<PipelineLayout>& pipeline_layout)
        : context(context), pipeline_layout(pipeline_layout) {}

    virtual ~Pipeline(){};

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

    void bind(const vk::CommandBuffer& cmd) {
        cmd.bindPipeline(get_pipeline_bind_point(), pipeline);
    }

    void bind_descriptor_set(const vk::CommandBuffer& cmd,
                                     const std::shared_ptr<DescriptorSet>& descriptor_set,
                                     const uint32_t first_set = 0) {
        cmd.bindDescriptorSets(get_pipeline_bind_point(), *pipeline_layout, first_set, 1,
                               &descriptor_set->get_set(), 0, nullptr);
    }

    // ---------------------------------------------------------------------------

    virtual vk::PipelineBindPoint get_pipeline_bind_point() const = 0;

    template <typename T>
    void push_constant(const vk::CommandBuffer& cmd, const T& constant, const uint32_t id = 0) {
        push_constant(cmd, reinterpret_cast<const void*>(&constant), id);
    }

    template <typename T>
    void push_constant(const vk::CommandBuffer& cmd, const T* constant, const uint32_t id = 0) {
        push_constant(cmd, reinterpret_cast<const void*>(constant), id);
    }

    // The id that was returned by the pipeline layout builder.
    void push_constant(const vk::CommandBuffer& cmd, const void* values, const uint32_t id = 0) {
        auto range = pipeline_layout->get_push_constant_range(id);
        push_constant(cmd, range.stageFlags, range.offset, range.size, values);
    }

    virtual void push_constant(const vk::CommandBuffer& cmd,
                               const vk::ShaderStageFlags flags,
                               const uint32_t offset,
                               const uint32_t size,
                               const void* values) {
        cmd.pushConstants(*pipeline_layout, flags, offset, size, values);
    }

  protected:
    const SharedContext context;
    const std::shared_ptr<PipelineLayout> pipeline_layout;
    vk::Pipeline pipeline;
    vk::PipelineBindPoint bind_point;
};

using PipelineHandle = std::shared_ptr<Pipeline>;

} // namespace merian
