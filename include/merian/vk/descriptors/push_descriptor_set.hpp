#pragma once

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/extension/extension_vk_descriptor_buffer.hpp"
#include "merian/vk/memory/memory_allocator.hpp"

#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

class PushDescriptorSet;
using PushDescriptorSetHandle = std::shared_ptr<PushDescriptorSet>;

// A PushDescriptorSet that knows its layout -> Can be used to simplify PushDescriptorSet updates.
// DescriptorsSet updates are queued until they are executed with a call to update(). In this
// case the update is performed immediately on the CPU timeline or update(cmd) for an update on the
// GPU timeline. The PushDescriptorSet holds references to the resources that are bound to it.
class PushDescriptorSet : public DescriptorContainer {

  public:
    // Allocates a PushDescriptorSet that matches the layout that is attached to the Pool
    PushDescriptorSet(const DescriptorSetLayoutHandle& layout)
        : DescriptorContainer(layout), context(layout->get_context()) {

        writes.resize(layout->get_descriptor_count());
        queued_writes.reserve(layout->get_descriptor_count());
    }

    ~PushDescriptorSet();

    const std::vector<vk::WriteDescriptorSet>& get_writes() {
        return writes;
    }    

    // ---------------------------------------------------------------------
    // Updates

    uint32_t update_count() const noexcept override {
        return queued_writes.size();
    }

    bool has_updates() const noexcept override {
        return !queued_writes.empty();
    }

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
            writes[get_layout()->get_binding_offset(write.dstBinding, write.dstArrayElement)] =
                write;
            apply_update_for(write.dstBinding, write.dstArrayElement);
        }

        queued_writes.clear();
    }

  protected:
    virtual void queue_write(vk::WriteDescriptorSet&& write) override {
        queued_writes.emplace_back(write);
    }

  public:
    static PushDescriptorSetHandle create(const DescriptorSetLayoutHandle& layout) {
        return PushDescriptorSetHandle(new PushDescriptorSet(layout));
    }

  private:
    const ContextHandle context;

    std::vector<vk::WriteDescriptorSet> queued_writes;
    std::vector<vk::WriteDescriptorSet> writes;
};

} // namespace merian
