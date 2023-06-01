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
    PipelineLayoutBuilder() {}

    PipelineLayoutBuilder& add_layout(vk::DescriptorSetLayout& layout) {
        layouts.push_back(layout);
        return *this;
    }

    PipelineLayoutBuilder& add_range(vk::PushConstantRange& range) {
        ranges.push_back(range);
        return *this;
    }

    PipelineLayoutBuilder& add_range(vk::ShaderStageFlags flags, uint32_t size, uint32_t offset = 0) {
        vk::PushConstantRange range{flags, offset, size};
        ranges.push_back(range);
        return *this;
    }

    template <typename T> PipelineLayoutBuilder& add_range(vk::ShaderStageFlags flags, uint32_t offset = 0) {
        add_range(flags, sizeof(T), offset);
        return *this;
    }

    vk::PipelineLayout build_layout(vk::Device& device, vk::PipelineLayoutCreateFlags flags = {}) {
        vk::PipelineLayoutCreateInfo info{flags, layouts, ranges};
        return device.createPipelineLayout(info);
    }

    vk::Pipeline build_compute(vk::Device& device,
                               vk::PipelineShaderStageCreateInfo stage_info,
                               vk::PipelineCreateFlags flags = {},
                               vk::PipelineLayoutCreateFlags layout_flags = {}) {
        vk::ComputePipelineCreateInfo info{flags, stage_info, build_layout(device, layout_flags)};
        return device.createComputePipeline(nullptr, info).value;
    }

  private:
    std::vector<vk::DescriptorSetLayout> layouts;
    std::vector<vk::PushConstantRange> ranges;
};

} // namespace merian
