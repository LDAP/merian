#pragma once

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/memory/memory_allocator.hpp"

#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

class DescriptorBuffer;
using DescriptorBufferHandle = std::shared_ptr<DescriptorBuffer>;

// A DescriptorBuffer that knows its layout -> Can be used to simplify DescriptorBuffer updates.
// DescriptorsBuffer updates are queued until they are executed with a call to update(). In this
// case the update is performed immediately on the CPU timeline or update(cmd) for an update on the
// GPU timeline. The DescriptorBuffer holds references to the resources that are bound to it.
class DescriptorBuffer : public DescriptorContainer {

  public:
    // Allocates a DescriptorBuffer that matches the layout that is attached to the Pool
    DescriptorBuffer(const DescriptorSetLayoutHandle& layout,
                     const MemoryAllocatorHandle& allocator)
        : DescriptorContainer(layout) {
        ext_descriptor_buffer =
            allocator->get_context()->get_extension<ExtensionVkDescriptorBuffer>();
        assert(ext_descriptor_buffer);

        const vk::BufferCreateInfo create_info{
            vk::BufferCreateFlags{}, layout->get_layout_size(),
            vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT |
                vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT};
        buffer = allocator->create_buffer(create_info, MemoryMappingType::HOST_ACCESS_RANDOM);
    }

    ~DescriptorBuffer() {}

    // ---------------------------------------------------------------------

    const BufferHandle& get_buffer() const {
        return buffer;
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
        vk::DescriptorAddressInfoEXT();
        // // for now descriptor count is always 1.
        // assert(writes.size() == write_resources.size());
        // assert(writes.size() == write_infos.size());

        // if (!has_updates()) {
        //     return;
        // }

        // for (uint32_t i = 0; i < writes.size(); i++) {
        //     vk::WriteDescriptorBuffer& write = writes[i];
        //     set_descriptor_info(write, write_infos[i]);

        //     // For now only descriptorCount 1 is implemented. Otherwise: If the dstBinding has
        //     // fewer than descriptorCount array elements remaining starting from
        //     // dstArrayElement, then the remainder will be used to update the subsequent binding
        //     // - dstBinding+1 starting at array element zero. In this case we'd have multiple
        //     // infos and resources i guess?
        //     assert(write.descriptorCount == 1);
        //     set_bindable_resource(std::move(write_resources[i]), write.dstBinding,
        //                           write.dstArrayElement);
        // }

        // pool->get_context()->device.updateDescriptorBuffers(writes, {});

        // writes.clear();
        // write_infos.clear();
        // write_resources.clear();
    }

    void update(const CommandBufferHandle& /*cmd*/) override {}

  protected:
    virtual void queue_write(vk::WriteDescriptorSet&& write) override {
        writes.emplace_back(write);
    }

  public:
    static DescriptorBufferHandle create(const DescriptorSetLayoutHandle& layout,
                                         const MemoryAllocatorHandle& allocator) {
        return DescriptorBufferHandle(new DescriptorBuffer(layout, allocator));
    }

  private:
    BufferHandle buffer;

    std::shared_ptr<ExtensionVkDescriptorBuffer> ext_descriptor_buffer;

    // ---------------------------------------------------------------------
    // Queued Updates

    // all with the same length.
    // Pointers are set in update() depending on the descriptor type.
    std::vector<vk::WriteDescriptorSet> writes;
    std::vector<BindableResourceHandle> write_resources;
    std::vector<std::variant<vk::DescriptorBufferInfo,
                             vk::DescriptorImageInfo,
                             vk::WriteDescriptorSetAccelerationStructureKHR>>
        write_infos;
};

} // namespace merian
