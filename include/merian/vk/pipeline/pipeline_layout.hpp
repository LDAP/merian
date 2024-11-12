#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include "vulkan/vulkan.hpp"
#include <spdlog/spdlog.h>

namespace merian {

class PipelineLayout : public std::enable_shared_from_this<PipelineLayout> {

  public:
    PipelineLayout(
        const ContextHandle& context,
        const std::vector<std::shared_ptr<DescriptorSetLayout>>& shared_descriptor_set_layouts,
        const std::vector<vk::PushConstantRange>& ranges = {},
        const vk::PipelineLayoutCreateFlags flags = {})
        : context(context), ranges(ranges),
          shared_descriptor_set_layouts(shared_descriptor_set_layouts) {
        SPDLOG_DEBUG("create PipelineLayout ({})", fmt::ptr(this));

        std::vector<vk::DescriptorSetLayout> descriptor_set_layouts(
            shared_descriptor_set_layouts.size());
        std::transform(shared_descriptor_set_layouts.begin(), shared_descriptor_set_layouts.end(),
                       descriptor_set_layouts.begin(),
                       [&](auto& shared) { return shared->get_layout(); });
        vk::PipelineLayoutCreateInfo info{flags, descriptor_set_layouts, ranges};
        pipeline_layout = context->device.createPipelineLayout(info);
    }

    ~PipelineLayout() {
        SPDLOG_DEBUG("destroy PipelineLayout ({})", fmt::ptr(this));
        context->device.destroyPipelineLayout(pipeline_layout);
    }

    operator const vk::PipelineLayout() const {
        return pipeline_layout;
    }

    const vk::PipelineLayout& get_pipeline_layout() {
        return pipeline_layout;
    }

    const ContextHandle& get_context() {
        return context;
    }

    const vk::PushConstantRange get_push_constant_range(uint32_t id) {
        assert(id < ranges.size() && "No such push constant. Did you declare a push constant?");
        return ranges[id];
    }

    const std::shared_ptr<DescriptorSetLayout>& get_descriptor_set_layout(const uint32_t set = 0) {
        assert(set < shared_descriptor_set_layouts.size());
        return shared_descriptor_set_layouts[set];
    }

  private:
    const ContextHandle context;
    const std::vector<vk::PushConstantRange> ranges;
    const std::vector<std::shared_ptr<DescriptorSetLayout>> shared_descriptor_set_layouts;
    vk::PipelineLayout pipeline_layout;
};

using PipelineLayoutHandle = std::shared_ptr<PipelineLayout>;

} // namespace merian
