#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include <map>
#include <vulkan/vulkan.hpp>

namespace merian {

class DescriptorPool : public std::enable_shared_from_this<DescriptorPool> {

  public:
    /**
     * @brief      Creates a DescriptorPool that has enough descriptors to allocate set_count DescriptorSet of the layout "layout".
     * 
     * By default the max_sets parameter is set to set_count
     */
    DescriptorPool(std::shared_ptr<DescriptorSetLayout>& layout,
                   const uint32_t set_count = 1,
                   const vk::DescriptorPoolCreateFlags flags = {},
                   const std::optional<uint32_t> max_sets = std::nullopt)
        : context(layout->get_context()), layout(layout), flags(flags) {
        SPDLOG_DEBUG("create DescriptorPool ({})", fmt::ptr(this));
        pool = descriptor_pool_for_bindings(layout->get_bindings(), layout->get_context()->device,
                                            set_count, flags, max_sets);
    }

    ~DescriptorPool() {
        SPDLOG_DEBUG("destroy DescriptorPool ({})", fmt::ptr(this));
        context->device.destroyDescriptorPool(pool);
    }

    operator const vk::DescriptorPool() const {
        return pool;
    }

    const vk::DescriptorPool& get_pool() const {
        return pool;
    }

    const std::shared_ptr<DescriptorSetLayout> get_layout() const {
        return layout;
    }

    const SharedContext get_context() const {
        return context;
    }

    const vk::DescriptorPoolCreateFlags get_create_flags() {
        return flags;
    }

    // // Returns all descriptor sets to the pool
    // void reset() {
    //     // Problem: DescriptorSets try to call free in their destuctor
    //     // Solution: Keep a weak reference of every set and notify the sets when they are destroyed
    // }

  private:
    const SharedContext context;
    const std::shared_ptr<DescriptorSetLayout> layout;
    const vk::DescriptorPoolCreateFlags flags;
    vk::DescriptorPool pool;

  public:
    // Determines the pool sizes to generate set_count DescriptorSets with these bindings.
    // E.g. set_count = 1 means with these sizes exactly once DescriptorSet with all bindings can be
    // created (or two with half of the bindings).
    static std::vector<vk::DescriptorPoolSize>
    make_pool_sizes_from_bindings(const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                                  uint32_t set_count = 1) {

        // We make one entry for each "type" of binding.
        std::vector<vk::DescriptorPoolSize> sizes;
        std::map<vk::DescriptorType, uint32_t> type_to_index;

        for (auto& binding : bindings) {
            if (!type_to_index.contains(binding.descriptorType)) {
                vk::DescriptorPoolSize size_for_type{binding.descriptorType,
                                                     binding.descriptorCount * set_count};
                type_to_index[binding.descriptorType] = sizes.size();
                sizes.push_back(size_for_type);
            } else {
                sizes[type_to_index[binding.descriptorType]].descriptorCount +=
                    binding.descriptorCount * set_count;
            }
        }

        return sizes;
    }

    // Creates a vk::DescriptorPool that has enough Descriptors such that set_count DescriptorSets
    // with these bindings can be created. By default the pools max sets property is set to
    // set_count, you can override that if this pool should be used for different layouts for
    // example.
    static vk::DescriptorPool
    descriptor_pool_for_bindings(const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                                 const vk::Device& device,
                                 const uint32_t set_count = 1,
                                 const vk::DescriptorPoolCreateFlags flags = {},
                                 const std::optional<uint32_t> max_sets = std::nullopt) {
        std::vector<vk::DescriptorPoolSize> pool_sizes =
            make_pool_sizes_from_bindings(bindings, set_count);
        vk::DescriptorPoolCreateInfo info{flags, max_sets.value_or(set_count), pool_sizes};
        return device.createDescriptorPool(info);
    }
};

} // namespace merian
