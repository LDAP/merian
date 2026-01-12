#pragma once

#include "merian/vk/context.hpp"
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

class DescriptorSetLayout : public std::enable_shared_from_this<DescriptorSetLayout> {
  public:
    static std::vector<vk::DescriptorPoolSize>
    pool_sizes_to_vector(const std::unordered_map<vk::DescriptorType, uint32_t>& pool_sizes,
                         const uint32_t multiplier) {
        std::vector<vk::DescriptorPoolSize> result;
        result.reserve(pool_sizes.size());
        for (const auto& size : pool_sizes) {
            result.emplace_back(size.first, size.second * multiplier);
        }
        return result;
    }

  public:
    DescriptorSetLayout(const ContextHandle& context,
                        const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                        const vk::DescriptorSetLayoutCreateFlags flags = {})
        : context(context), bindings(bindings), binding_offsets(bindings.size(), 0) {
        vk::DescriptorSetLayoutCreateInfo info{flags, bindings};
        SPDLOG_DEBUG("create DescriptorSetLayout ({})", fmt::ptr(this));
        layout = context->device.createDescriptorSetLayout(info);

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

    uint32_t get_descriptor_count() const noexcept {
        return descriptor_count;
    }

    // returns the offset into an array where all bindings are linearized.
    uint32_t get_binding_offset(const uint32_t binding, const uint32_t array_element = 0) const {
        assert(binding < binding_offsets.size());
        assert(array_element < bindings[binding].descriptorCount);
        return binding_offsets[binding] + array_element;
    }

    const std::unordered_map<vk::DescriptorType, uint32_t>& get_pool_sizes() {
        return pool_sizes;
    }

    std::vector<vk::DescriptorPoolSize> get_pool_sizes_as_vector(const uint32_t multiplier = 1) {
        return pool_sizes_to_vector(pool_sizes, multiplier);
    }

  private:
    const ContextHandle context;
    const std::vector<vk::DescriptorSetLayoutBinding> bindings;
    std::unordered_map<vk::DescriptorType, uint32_t> pool_sizes;
    vk::DescriptorSetLayout layout;

    uint32_t descriptor_count = 0;
    std::vector<uint32_t> binding_offsets;
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
