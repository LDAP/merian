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
                         const uint32_t multiplier);

  public:
    DescriptorSetLayout(const ContextHandle& context,
                        const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                        const vk::DescriptorSetLayoutCreateFlags flags = {});

    ~DescriptorSetLayout();

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
    uint32_t get_binding_offset(const uint32_t binding, const uint32_t array_element = 0) const;

    const std::unordered_map<vk::DescriptorType, uint32_t>& get_pool_sizes() {
        return pool_sizes;
    }

    std::vector<vk::DescriptorPoolSize> get_pool_sizes_as_vector(const uint32_t multiplier = 1) {
        return pool_sizes_to_vector(pool_sizes, multiplier);
    }

    bool supports_descriptor_buffer() const {
        return bool(flags & vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT);
    }

    bool supports_descriptor_set() const {
        // https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorSetLayoutCreateFlagBits.html#
        return !supports_descriptor_buffer();
    }

  private:
    const ContextHandle context;
    const std::vector<vk::DescriptorSetLayoutBinding> bindings;
    const vk::DescriptorSetLayoutCreateFlags flags;

    std::unordered_map<vk::DescriptorType, uint32_t> pool_sizes;
    vk::DescriptorSetLayout layout;

    uint32_t descriptor_count = 0;
    std::vector<uint32_t> binding_offsets;
};
using DescriptorSetLayoutHandle = std::shared_ptr<DescriptorSetLayout>;

std::string format_as(const DescriptorSetLayoutHandle& layout);

} // namespace merian
