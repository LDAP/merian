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

    PipelineLayoutBuilder&
    add_descriptor_set_layout(const std::shared_ptr<DescriptorSetLayout>& layout) {
        assert(layout);
        shared_descriptor_set_layouts.push_back(layout);
        return *this;
    }

    // Adds a push constant. The constant can be set on the pipeline using the push_constant_id.
    // The id of the first is 0, the second is 1,...
    // You can get the id by supplying a pointer to push_constant_id;
    template <typename T>
    PipelineLayoutBuilder&
    add_push_constant(const vk::ShaderStageFlags flags = vk::ShaderStageFlagBits::eCompute,
                      uint32_t* push_constant_id = nullptr) {
        return add_push_constant(sizeof(T), flags, push_constant_id);
    }

    PipelineLayoutBuilder&
    add_push_constant(uint32_t size,
                      const vk::ShaderStageFlags flags = vk::ShaderStageFlagBits::eCompute,
                      uint32_t* push_constant_id = nullptr) {
        vk::PushConstantRange range{flags, current_push_constant_offset, size};
        current_push_constant_offset += size;
        if (push_constant_id) {
            *push_constant_id = ranges.size();
        }
        ranges.push_back(range);
        return *this;
    }

    std::shared_ptr<PipelineLayout>
    build_pipeline_layout(const vk::PipelineLayoutCreateFlags flags = {}) {
        return std::make_shared<PipelineLayout>(context, shared_descriptor_set_layouts, ranges,
                                                flags);
    }

  private:
    const SharedContext context;
    std::vector<std::shared_ptr<DescriptorSetLayout>> shared_descriptor_set_layouts;
    std::vector<vk::PushConstantRange> ranges;
    uint32_t current_push_constant_offset = 0;
};

} // namespace merian
