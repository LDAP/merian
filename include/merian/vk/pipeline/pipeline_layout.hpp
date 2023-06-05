#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include "vulkan/vulkan.hpp"
#include <spdlog/spdlog.h>

namespace merian {

class PipelineLayout : public std::enable_shared_from_this<PipelineLayout> {

  public:
    PipelineLayout(
        const SharedContext& context,
        const std::vector<std::shared_ptr<DescriptorSetLayout>>& shared_descriptor_set_layouts,
        const std::vector<vk::PushConstantRange>& ranges = {},
        const vk::PipelineLayoutCreateFlags flags = {})
        : context(context), shared_descriptor_set_layouts(shared_descriptor_set_layouts) {
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

    const vk::PipelineLayout& get_pipeline() {
        return pipeline_layout;
    }

    const SharedContext& get_context() {
        return context;
    }

  private:
    const SharedContext context;
    const std::vector<std::shared_ptr<DescriptorSetLayout>> shared_descriptor_set_layouts;
    vk::PipelineLayout pipeline_layout;
};

} // namespace merian
