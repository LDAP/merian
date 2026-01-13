#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include "vulkan/vulkan.hpp"
#include <spdlog/spdlog.h>

namespace merian {

class PipelineLayout : public std::enable_shared_from_this<PipelineLayout> {

  public:
    PipelineLayout(const ContextHandle& context,
                   const std::vector<std::shared_ptr<DescriptorSetLayout>>& descriptor_set_layouts,
                   const std::vector<vk::PushConstantRange>& ranges = {},
                   const vk::PipelineLayoutCreateFlags flags = {})
        : context(context), ranges(ranges), descriptor_set_layouts(descriptor_set_layouts),
          flags(flags) {
        SPDLOG_DEBUG("create PipelineLayout ({})", fmt::ptr(this));

        std::vector<vk::DescriptorSetLayout> vk_descriptor_set_layouts(
            descriptor_set_layouts.size());
        std::transform(descriptor_set_layouts.begin(), descriptor_set_layouts.end(),
                       vk_descriptor_set_layouts.begin(),
                       [&](auto& shared) { return shared->get_layout(); });
        vk::PipelineLayoutCreateInfo info{flags, vk_descriptor_set_layouts, ranges};
        pipeline_layout = context->device.createPipelineLayout(info);
    }

    ~PipelineLayout() {
        SPDLOG_DEBUG("destroy PipelineLayout ({})", fmt::ptr(this));
        context->device.destroyPipelineLayout(pipeline_layout);
    }

    operator const vk::PipelineLayout&() const {
        return pipeline_layout;
    }

    const vk::PipelineLayout& get_pipeline_layout() const {
        return pipeline_layout;
    }

    const ContextHandle& get_context() const {
        return context;
    }

    const vk::PushConstantRange& get_push_constant_range(uint32_t id) const {
        assert(id < ranges.size() && "No such push constant. Did you declare a push constant?");
        return ranges[id];
    }

    const std::shared_ptr<DescriptorSetLayout>&
    get_descriptor_set_layout(const uint32_t set = 0) const {
        assert(set < descriptor_set_layouts.size());
        return descriptor_set_layouts[set];
    }

  private:
    const ContextHandle context;
    const std::vector<vk::PushConstantRange> ranges;
    const std::vector<DescriptorSetLayoutHandle> descriptor_set_layouts;
    const vk::PipelineLayoutCreateFlags flags;

    vk::PipelineLayout pipeline_layout;
};

using PipelineLayoutHandle = std::shared_ptr<PipelineLayout>;

} // namespace merian
