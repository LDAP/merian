#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_vk_descriptor_buffer.hpp"
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

class DescriptorSetLayout : public std::enable_shared_from_this<DescriptorSetLayout> {

  public:
    DescriptorSetLayout(const ContextHandle& context,
                        const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                        const vk::DescriptorSetLayoutCreateFlags flags = {})
        : context(context), bindings(bindings) {
        vk::DescriptorSetLayoutCreateInfo info{flags, bindings};
        SPDLOG_DEBUG("create DescriptorSetLayout ({})", fmt::ptr(this));
        layout = context->device.createDescriptorSetLayout(info);
    }

    ~DescriptorSetLayout() {
        SPDLOG_DEBUG("destroy DescriptorSetLayout ({})", fmt::ptr(this));
        context->device.destroyDescriptorSetLayout(layout);
    }

    operator const vk::DescriptorSetLayout&() const {
        return layout;
    }

    const vk::DescriptorSetLayout& get_layout() const {
        return layout;
    }

    const std::vector<vk::DescriptorSetLayoutBinding>& get_bindings() const {
        return bindings;
    }

    const ContextHandle& get_context() const {
        return context;
    }

    vk::DescriptorType get_type_for_binding(uint32_t binding) const {
        return bindings[binding].descriptorType;
    }

    // size in bytes for a descriptor buffer.
    vk::DeviceSize get_layout_size() {
        assert(context->get_extension<ExtensionVkDescriptorBuffer>());
        return context->device.getDescriptorSetLayoutSizeEXT(layout);
    }

    vk::DeviceSize get_layout_binding_offset(const uint32_t binding,
                                             const uint32_t array_element = 0) {
        assert(context->get_extension<ExtensionVkDescriptorBuffer>());
        return context->device.getDescriptorSetLayoutBindingOffsetEXT(layout, binding) +
               (array_element *
                context->get_extension<ExtensionVkDescriptorBuffer>()->descriptor_size_for_type(
                    get_type_for_binding(binding)));
    }

  private:
    const ContextHandle context;
    const std::vector<vk::DescriptorSetLayoutBinding> bindings;
    vk::DescriptorSetLayout layout;
};
using DescriptorSetLayoutHandle = std::shared_ptr<DescriptorSetLayout>;

inline std::string format_as(const DescriptorSetLayoutHandle& layout) {
    if (layout->get_bindings().empty())
        return "empty";

    std::vector<std::string> binding_strs;

    for (const auto& binding : layout->get_bindings()) {
        binding_strs.emplace_back(
            fmt::format("(binding = {}, count = {}, type = {}, stage flags = {})", binding.binding,
                        binding.descriptorCount, vk::to_string(binding.descriptorType),
                        vk::to_string(binding.stageFlags)));
    }

    return fmt::format("{}", fmt::join(binding_strs.begin(), binding_strs.end(), ",\n"));
}

} // namespace merian
