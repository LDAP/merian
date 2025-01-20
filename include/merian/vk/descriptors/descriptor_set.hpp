#pragma once

#include "merian/vk/descriptors/descriptor_pool.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

// Allocates one set for each layout
inline std::vector<vk::DescriptorSet>
allocate_descriptor_sets(const vk::Device& device,
                         const vk::DescriptorPool& pool,
                         const std::vector<vk::DescriptorSetLayout>& layouts) {
    vk::DescriptorSetAllocateInfo info{pool, layouts};
    return device.allocateDescriptorSets(info);
}

// Allocates `count` sets for the supplied layout.
inline std::vector<vk::DescriptorSet>
allocate_descriptor_sets(const vk::Device& device,
                         const vk::DescriptorPool& pool,
                         const vk::DescriptorSetLayout& layout,
                         uint32_t count) {
    std::vector<vk::DescriptorSetLayout> layouts(count, layout);
    return allocate_descriptor_sets(device, pool, layouts);
}

inline vk::DescriptorSet allocate_descriptor_set(const vk::Device& device,
                                                 const vk::DescriptorPool& pool,
                                                 const vk::DescriptorSetLayout& layout) {
    return allocate_descriptor_sets(device, pool, {layout})[0];
}

// A DescriptorSet that knows its layout -> Can be used to simplify DescriptorSetUpdate.
class DescriptorSet : public std::enable_shared_from_this<DescriptorSet>, public Object {

  public:
    // Allocates a DescriptorSet that matches the layout that is attached to the Pool
    DescriptorSet(const std::shared_ptr<DescriptorPool>& pool)
        : pool(pool), layout(pool->get_layout()) {
        SPDLOG_DEBUG("allocating DescriptorSet ({})", fmt::ptr(this));
        set = allocate_descriptor_set(*pool->get_context(), *pool, *pool->get_layout());
    }

    ~DescriptorSet() {
        if (pool->get_create_flags() & vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet) {
            // DescriptorSet can be given back to the DescriptorPool
            SPDLOG_DEBUG("freeing DescriptorSet ({})", fmt::ptr(this));
            pool->get_context()->device.freeDescriptorSets(*pool, {set});
        } else {
            SPDLOG_DEBUG("destroying DescriptorSet ({}) but not freeing since the pool was not "
                         "created with the {} bit set.",
                         fmt::ptr(this),
                         vk::to_string(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet));
        }
    }

    operator const vk::DescriptorSet&() const {
        return set;
    }

    operator const vk::DescriptorSetLayout&() const {
        return *layout;
    }

    const vk::DescriptorSet& operator*() const {
        return set;
    }

    const vk::DescriptorSet& get_descriptor_set() const {
        return set;
    }

    const DescriptorSetLayoutHandle& get_layout() const {
        return layout;
    }

    vk::DescriptorType get_type_for_binding(uint32_t binding) const {
        return layout->get_bindings()[binding].descriptorType;
    }

  private:
    const std::shared_ptr<DescriptorPool> pool;
    const std::shared_ptr<DescriptorSetLayout> layout;
    vk::DescriptorSet set;
};

using DescriptorSetHandle = std::shared_ptr<DescriptorSet>;

} // namespace merian
