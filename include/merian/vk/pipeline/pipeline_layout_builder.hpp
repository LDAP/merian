#pragma once

#include "merian/vk/descriptors/descriptor_set_layout.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"

#include <memory>
#include <vulkan/vulkan.hpp>

namespace merian {

/**
 * @brief      Builder for PipelineLayouts.
 *
 * E.g. to define a push constant:
 *
 * struct MyPushConstant {
 *      uint32_t constant_one;
 *      uint32_t constant_two;
 * }
 *
 * auto pipeline_layout = PipelineLayoutBuilder()
 *      .add_range<MyPushConstant>(vk::ShaderStageFlagBits::eCompute)
 *      .add_layout(descriptor_set_layout)
 *      .build_layout();
 *
 */
class PipelineLayoutBuilder {

  public:
    PipelineLayoutBuilder(const SharedContext& context) : context(context) {}

    PipelineLayoutBuilder& add_descriptor_set_layout(const std::shared_ptr<DescriptorSetLayout>& layout) {
        shared_descriptor_set_layouts.push_back(layout);
        return *this;
    }

    PipelineLayoutBuilder& add_range(const vk::PushConstantRange& range) {
        ranges.push_back(range);
        return *this;
    }

    PipelineLayoutBuilder&
    add_range(const vk::ShaderStageFlags flags, const uint32_t size, const uint32_t offset = 0) {
        vk::PushConstantRange range{flags, offset, size};
        ranges.push_back(range);
        return *this;
    }

    template <typename T>
    PipelineLayoutBuilder& add_range(const vk::ShaderStageFlags flags, const uint32_t offset = 0) {
        add_range(flags, sizeof(T), offset);
        return *this;
    }

    std::shared_ptr<PipelineLayout> build_pipeline_layout(const vk::PipelineLayoutCreateFlags flags = {}) {
        return std::make_shared<PipelineLayout>(context, shared_descriptor_set_layouts, ranges, flags);
    }

  private:
    const SharedContext context;
    std::vector<std::shared_ptr<DescriptorSetLayout>> shared_descriptor_set_layouts;
    std::vector<vk::PushConstantRange> ranges;
};

} // namespace merian
