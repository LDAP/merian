#include "merian/vk/descriptors/descriptor_buffer.hpp"

#include "merian/utils/pointer.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/utils/barriers.hpp"

namespace merian {

void DescriptorBuffer::make_desc_get_info(vk::DescriptorGetInfoEXT& desc_get_info,
                                          vk::DescriptorAddressInfoEXT& address_info,
                                          const vk::WriteDescriptorSet& write) {
    switch (write.descriptorType) {
    case vk::DescriptorType::eStorageBuffer:
    case vk::DescriptorType::eUniformBuffer: {
        assert(write.pBufferInfo);
        const BufferHandle buffer = debugable_ptr_cast<Buffer>(
            get_bindable_resource_at(write.dstBinding, write.dstArrayElement));
        address_info = buffer->get_descriptor_address_info(write.pBufferInfo->offset,
                                                           write.pBufferInfo->range);
        desc_get_info = {write.descriptorType, vk::DescriptorDataEXT(&address_info)};
        return;
    }
    case vk::DescriptorType::eCombinedImageSampler:
    case vk::DescriptorType::eSampledImage:
    case vk::DescriptorType::eStorageImage: {
        assert(write.pImageInfo);
        desc_get_info = {write.descriptorType, vk::DescriptorDataEXT(write.pImageInfo)};
        return;
    }
    case vk::DescriptorType::eUniformTexelBuffer:
    case vk::DescriptorType::eStorageTexelBuffer: {
        assert(write.pTexelBufferView);
        throw std::runtime_error{"not implemented"};
    }
    case vk::DescriptorType::eAccelerationStructureKHR:
    case vk::DescriptorType::eAccelerationStructureNV: {
        assert(write.pNext);
        const AccelerationStructureHandle as = debugable_ptr_cast<AccelerationStructure>(
            get_bindable_resource_at(write.dstBinding, write.dstArrayElement));
        desc_get_info = {write.descriptorType,
                         vk::DescriptorDataEXT(as->get_acceleration_structure_device_address())};
        return;
    }
    default:
        throw std::runtime_error{"unsupported descriptor type"};
    }
}

DescriptorBuffer::~DescriptorBuffer() {
    delete[] scratch;
}

void DescriptorBuffer::update() {

    std::byte* gpu_buffer = buffer->get_memory()->map_as<std::byte>();

    for (uint32_t i = 0; i < queued_writes.size(); i++) {

        vk::WriteDescriptorSet& write = queued_writes[i];
        assert(write.descriptorCount == 1);

        // we need to access the new resource below.
        apply_update_for(write.dstBinding, write.dstArrayElement);

        const std::size_t size = binding_infos[write.dstBinding].size;
        const vk::DeviceSize offset =
            get_layout_binding_offset(write.dstBinding, write.dstArrayElement);

        vk::DescriptorGetInfoEXT desc_get_info;
        vk::DescriptorAddressInfoEXT address_info;

        make_desc_get_info(desc_get_info, address_info, write);

        context->get_device()->get_device().getDescriptorEXT(&desc_get_info, size, gpu_buffer + offset);
    }

    buffer->get_memory()->unmap();
    queued_writes.clear();
    // we get a implicit host -> device sync on the next queue submit.
}

void DescriptorBuffer::update(const CommandBufferHandle& cmd) {
    cmd->barrier(buffer->buffer_barrier2(all_shaders2, vk::PipelineStageFlagBits2::eTransfer,
                                         vk::AccessFlagBits2::eDescriptorBufferReadEXT,
                                         vk::AccessFlagBits2::eTransferWrite));

    for (uint32_t i = 0; i < queued_writes.size(); i++) {

        vk::WriteDescriptorSet& write = queued_writes[i];
        assert(write.descriptorCount == 1);

        // we need to access the new resource below.
        apply_update_for(write.dstBinding, write.dstArrayElement);

        const std::size_t size = binding_infos[write.dstBinding].size;
        const vk::DeviceSize offset =
            get_layout_binding_offset(write.dstBinding, write.dstArrayElement);

        vk::DescriptorGetInfoEXT desc_get_info;
        vk::DescriptorAddressInfoEXT address_info;

        make_desc_get_info(desc_get_info, address_info, write);

        context->get_device()->get_device().getDescriptorEXT(&desc_get_info, size, scratch);
        cmd->update(buffer, offset, size, scratch);
    }
    queued_writes.clear();

    cmd->barrier(buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer, all_shaders2,
                                         vk::AccessFlagBits2::eTransferWrite,
                                         vk::AccessFlagBits2::eDescriptorBufferReadEXT));
}

void DescriptorBuffer::bind(const CommandBufferHandle& cmd,
                            const PipelineHandle& pipeline,
                            const uint32_t descriptor_set_index) const {
    cmd->bind_descriptor_buffer(pipeline, descriptor_set_index,
                                std::static_pointer_cast<const DescriptorBuffer>(shared_from_this()));
}

} // namespace merian
