#pragma once

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/descriptors/descriptor_pool.hpp"

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

class DescriptorSet;
using DescriptorSetHandle = std::shared_ptr<DescriptorSet>;

// A DescriptorSet that knows its layout -> Can be used to simplify DescriptorSet updates.
// DescriptorsSet updates are queued until they are executed with a call to update(). In this case
// the update is performed immediately.
// The DescriptorSet holds references to the resources that are bound to it.
class DescriptorSet : public DescriptorContainer {

  public:
    // Allocates a DescriptorSet that matches the layout that is attached to the Pool
    DescriptorSet(const DescriptorPoolHandle& pool)
        : DescriptorContainer(pool->get_layout()), pool(pool) {
        set = allocate_descriptor_set(*pool->get_context(), *pool, *pool->get_layout());
        SPDLOG_DEBUG("allocated DescriptorSet ({})", fmt::ptr(static_cast<VkDescriptorSet>(set)));

        writes.reserve(get_layout()->get_descriptor_count());
    }

    ~DescriptorSet() {
        if (pool->get_create_flags() & vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet) {
            // DescriptorSet can be given back to the DescriptorPool
            SPDLOG_DEBUG("freeing DescriptorSet ({})", fmt::ptr(static_cast<VkDescriptorSet>(set)));
            pool->get_context()->device.freeDescriptorSets(*pool, set);
        } else {
            SPDLOG_DEBUG("destroying DescriptorSet ({}) but not freeing since the pool was not "
                         "created with the {} bit set.",
                         fmt::ptr(static_cast<VkDescriptorSet>(set)),
                         vk::to_string(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet));
        }
    }

    operator const vk::DescriptorSet&() const {
        return set;
    }

    const vk::DescriptorSet& operator*() const {
        return set;
    }

    const vk::DescriptorSet& get_descriptor_set() const {
        return set;
    }

    // ---------------------------------------------------------------------
    // Updates

    uint32_t update_count() const noexcept override {
        return writes.size();
    }

    bool has_updates() const noexcept override {
        return !writes.empty();
    }

    void update() override {
        if (!has_updates()) {
            return;
        }

        for (uint32_t i = 0; i < writes.size(); i++) {
            vk::WriteDescriptorSet& write = writes[i];

            // For now only descriptorCount 1 is implemented. Otherwise: If the dstBinding has
            // fewer than descriptorCount array elements remaining starting from
            // dstArrayElement, then the remainder will be used to update the subsequent binding
            // - dstBinding+1 starting at array element zero. In this case we'd have multiple
            // infos and resources i guess?
            assert(write.descriptorCount == 1);
            apply_update_for(write.dstBinding, write.dstArrayElement);
        }

        pool->get_context()->device.updateDescriptorSets(writes, {});

        writes.clear();
    }

  protected:
    virtual void queue_write(vk::WriteDescriptorSet&& write) override {
        writes.emplace_back(write);
        writes.back().dstSet = set;
    }

  public:
    static DescriptorSetHandle create(const DescriptorPoolHandle& pool) {
        return DescriptorSetHandle(new DescriptorSet(pool));
    }

  private:
    const DescriptorPoolHandle pool;
    vk::DescriptorSet set;

    // ---------------------------------------------------------------------
    // Queued Updates

    std::vector<vk::WriteDescriptorSet> writes;
};

} // namespace merian
