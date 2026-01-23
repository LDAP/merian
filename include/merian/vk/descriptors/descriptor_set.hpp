#pragma once

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/descriptors/descriptor_pool.hpp"

#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

class DescriptorSet;
using DescriptorSetHandle = std::shared_ptr<DescriptorSet>;
using ConstDescriptorSetHandle = std::shared_ptr<const DescriptorSet>;

// A DescriptorSet that knows its layout -> Can be used to simplify DescriptorSet updates.
// DescriptorsSet updates are queued until they are executed with a call to update(). In this case
// the update is performed immediately.
// The DescriptorSet holds references to the resources that are bound to it.
class DescriptorSet : public DescriptorContainer {

    friend class DescriptorPool;

  private:
    DescriptorSet(const DescriptorPoolHandle& pool,
                  const DescriptorSetLayoutHandle& layout,
                  const vk::DescriptorSet& set)
        : DescriptorContainer(layout), pool(pool), set(set) {
        SPDLOG_DEBUG("allocated DescriptorSet ({})", fmt::ptr(static_cast<VkDescriptorSet>(set)));

        queued_writes.reserve(get_layout()->get_descriptor_count());
    }

  public:
    ~DescriptorSet() {
        pool->free(this);
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
        return queued_writes.size();
    }

    bool has_updates() const noexcept override {
        return !queued_writes.empty();
    }

    void bind(const CommandBufferHandle& cmd,
              const PipelineHandle& pipeline,
              const uint32_t descriptor_set_index) const override;

    void update() override {
        if (!has_updates()) {
            return;
        }

        for (uint32_t i = 0; i < queued_writes.size(); i++) {
            vk::WriteDescriptorSet& write = queued_writes[i];

            // For now only descriptorCount 1 is implemented. Otherwise: If the dstBinding has
            // fewer than descriptorCount array elements remaining starting from
            // dstArrayElement, then the remainder will be used to update the subsequent binding
            // - dstBinding+1 starting at array element zero. In this case we'd have multiple
            // infos and resources i guess?
            assert(write.descriptorCount == 1);
            apply_update_for(write.dstBinding, write.dstArrayElement);
        }

        pool->get_context()->get_device()->get_device().updateDescriptorSets(queued_writes, {});

        queued_writes.clear();
    }

  protected:
    virtual void queue_write(vk::WriteDescriptorSet&& write) override {
        queued_writes.emplace_back(write);
        queued_writes.back().dstSet = set;
    }

  private:
    static DescriptorSetHandle create(const DescriptorPoolHandle& pool,
                                      const DescriptorSetLayoutHandle& layout,
                                      const vk::DescriptorSet& set) {
        return DescriptorSetHandle(new DescriptorSet(pool, layout, set));
    }

  private:
    const DescriptorPoolHandle pool;
    const vk::DescriptorSet set;

    // ---------------------------------------------------------------------
    // Queued Updates

    std::vector<vk::WriteDescriptorSet> queued_writes;
};

} // namespace merian
