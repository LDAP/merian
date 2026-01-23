#pragma once

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/extension/extension_vk_descriptor_buffer.hpp"
#include "merian/vk/memory/memory_allocator.hpp"

#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

class DescriptorBuffer;
using DescriptorBufferHandle = std::shared_ptr<DescriptorBuffer>;
using ConstDescriptorBufferHandle = std::shared_ptr<const DescriptorBuffer>;

// A DescriptorBuffer that knows its layout -> Can be used to simplify DescriptorBuffer updates.
// DescriptorsBuffer updates are queued until they are executed with a call to update(). In this
// case the update is performed immediately on the CPU timeline or update(cmd) for an update on the
// GPU timeline. The DescriptorBuffer holds references to the resources that are bound to it.
class DescriptorBuffer : public DescriptorContainer {

  public:
    // Allocates a DescriptorBuffer that matches the layout that is attached to the Pool
    DescriptorBuffer(const DescriptorSetLayoutHandle& layout,
                     const MemoryAllocatorHandle& allocator)
        : DescriptorContainer(layout), context(layout->get_context()) {
        assert(layout->supports_descriptor_buffer());

        const auto ext_descriptor_buffer =
            allocator->get_context()->get_extension<ExtensionVkDescriptorBuffer>();
        assert(ext_descriptor_buffer);

        const vk::BufferCreateInfo create_info{
            vk::BufferCreateFlags{}, get_size(),
            vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT |
                vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT |
                vk::BufferUsageFlagBits::eShaderDeviceAddress};
        buffer =
            allocator->create_buffer(create_info, MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE);

        binding_infos.resize(layout->get_bindings().size());
        vk::DeviceSize max_binding_size = 0;
        for (uint32_t i = 0; i < layout->get_bindings().size(); i++) {
            auto& binding_info = binding_infos[i];

            binding_info.size =
                ext_descriptor_buffer->descriptor_size_for_type(layout->get_type_for_binding(i));
            binding_info.offset =
                context->get_device()->get_device().getDescriptorSetLayoutBindingOffsetEXT(*get_layout(), i);

            max_binding_size = std::max(max_binding_size, binding_info.size);
        }

        queued_writes.reserve(layout->get_descriptor_count());
        scratch = new std::byte[max_binding_size];
    }

    ~DescriptorBuffer();

    // ---------------------------------------------------------------------

    const BufferHandle& get_buffer() const {
        return buffer;
    }

    // size in bytes for a descriptor buffer.
    vk::DeviceSize get_size() {
        return context->get_device()->get_device().getDescriptorSetLayoutSizeEXT(*get_layout());
    }

    vk::DeviceSize get_layout_binding_offset(const uint32_t binding,
                                             const uint32_t array_element = 0) {
        assert(binding < binding_infos.size());
        return binding_infos[binding].offset + (array_element * binding_infos[binding].size);
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

    void update() override;

    void update(const CommandBufferHandle& cmd);

  protected:
    virtual void queue_write(vk::WriteDescriptorSet&& write) override {
        queued_writes.emplace_back(write);
    }

  private:
    void make_desc_get_info(vk::DescriptorGetInfoEXT& desc_get_info,
                            vk::DescriptorAddressInfoEXT& info,
                            const vk::WriteDescriptorSet& write);

  public:
    static DescriptorBufferHandle create(const DescriptorSetLayoutHandle& layout,
                                         const MemoryAllocatorHandle& allocator) {
        return DescriptorBufferHandle(new DescriptorBuffer(layout, allocator));
    }

  private:
    const ContextHandle context;

    BufferHandle buffer;

    struct BindingInfo {
        vk::DeviceSize size;
        vk::DeviceSize offset;
    };

    std::vector<BindingInfo> binding_infos;
    std::byte* scratch;
    std::vector<vk::WriteDescriptorSet> queued_writes;
};

} // namespace merian
