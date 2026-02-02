#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

namespace merian {

std::vector<vk::DescriptorPoolSize>
DescriptorSetLayout::pool_sizes_to_vector(const std::unordered_map<vk::DescriptorType, uint32_t>& pool_sizes,
                                          const uint32_t multiplier) {
    std::vector<vk::DescriptorPoolSize> result;
    result.reserve(pool_sizes.size());
    for (const auto& size : pool_sizes) {
        result.emplace_back(size.first, size.second * multiplier);
    }
    return result;
}

DescriptorSetLayout::DescriptorSetLayout(const ContextHandle& context,
                                         const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                                         const vk::DescriptorSetLayoutCreateFlags flags)
    : context(context), bindings(bindings), flags(flags), binding_offsets(bindings.size(), 0) {
    vk::DescriptorSetLayoutCreateInfo info{flags, bindings};
    SPDLOG_DEBUG("create DescriptorSetLayout ({})", fmt::ptr(this));
    layout = context->get_device()->get_device().createDescriptorSetLayout(info);

    if (!bindings.empty()) {
        for (uint32_t i = 1; i < bindings.size(); i++) {
            binding_offsets[i] = bindings[i - 1].descriptorCount + binding_offsets[i - 1];
        }
        descriptor_count = binding_offsets.back() + bindings.back().descriptorCount;
    }

    for (const auto& binding : bindings) {
        pool_sizes[binding.descriptorType] += binding.descriptorCount;
    }
}

DescriptorSetLayout::~DescriptorSetLayout() {
    SPDLOG_DEBUG("destroy DescriptorSetLayout ({})", fmt::ptr(this));
    context->get_device()->get_device().destroyDescriptorSetLayout(layout);
}

uint32_t DescriptorSetLayout::get_binding_offset(const uint32_t binding, const uint32_t array_element) const {
    assert(binding < binding_offsets.size());
    assert(array_element < bindings[binding].descriptorCount);
    return binding_offsets[binding] + array_element;
}

std::string format_as(const DescriptorSetLayoutHandle& layout) {
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
