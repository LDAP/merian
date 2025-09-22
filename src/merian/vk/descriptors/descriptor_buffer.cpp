#include "merian/vk/descriptors/descriptor_buffer.hpp"

#include "merian/utils/pointer.hpp"
#include "merian/vk/command/command_buffer.hpp"

namespace merian {

void DescriptorBuffer::update() {

    std::byte* gpu_buffer = buffer->get_memory()->map_as<std::byte>();

    // for (uint32_t i = 0; i < writes.size(); i++) {
    //     const QueueElement& write = writes[i];
    //     apply_update_for(write.binding, write.array_element);
    //     memcpy(gpu_buffer + write.offset, host_buffer.data() + write.offset, write.size);
    // }

    for (uint32_t i = 0; i < writes.size(); i++) {

        vk::WriteDescriptorSet& write = writes[i];
        assert(write.descriptorCount == 1);

        // we need to access the new resource below.
        apply_update_for(write.dstBinding, write.dstArrayElement);

        switch (write.descriptorType) {
        case vk::DescriptorType::eStorageBuffer:
        case vk::DescriptorType::eUniformBuffer: {
            assert(write.pBufferInfo);
            const BufferHandle buffer = debugable_ptr_cast<Buffer>(
                get_bindable_resource_at(write.dstBinding, write.dstArrayElement));
            const vk::DescriptorAddressInfoEXT info = buffer->get_descriptor_address_info(
                write.pBufferInfo->offset, write.pBufferInfo->range);
            const vk::DescriptorGetInfoEXT desc_get_info(write.descriptorType,
                                                         vk::DescriptorDataEXT(&info));
            const std::size_t size = binding_infos[write.dstBinding].size;
            const vk::DeviceSize offset =
                get_layout_binding_offset(write.dstBinding, write.dstArrayElement);
            context->device.getDescriptorEXT(&desc_get_info, size, gpu_buffer + offset);
            break;
        }
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage: {
            assert(write.pImageInfo);
            const vk::DescriptorGetInfoEXT desc_get_info(write.descriptorType,
                                                         vk::DescriptorDataEXT(write.pImageInfo));
            const std::size_t size = binding_infos[write.dstBinding].size;
            const vk::DeviceSize offset =
                get_layout_binding_offset(write.dstBinding, write.dstArrayElement);

            context->device.getDescriptorEXT(&desc_get_info, size, gpu_buffer + offset);
            break;
        }
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer: {
            assert(write.pTexelBufferView);
            throw std::runtime_error{"not implemented"};
        }
        case vk::DescriptorType::eAccelerationStructureKHR:
        case vk::DescriptorType::eAccelerationStructureNV: {
            const AccelerationStructureHandle as = debugable_ptr_cast<AccelerationStructure>(
                get_bindable_resource_at(write.dstBinding, write.dstArrayElement));
            const vk::DescriptorGetInfoEXT desc_get_info(
                write.descriptorType,
                vk::DescriptorDataEXT(as->get_acceleration_structure_device_address()));
            const std::size_t size = binding_infos[write.dstBinding].size;
            const vk::DeviceSize offset =
                get_layout_binding_offset(write.dstBinding, write.dstArrayElement);

            context->device.getDescriptorEXT(&desc_get_info, size, gpu_buffer + offset);
            break;
        }
        default:
            throw std::runtime_error{"unsupported descriptor type"};
        }
    }

    buffer->get_memory()->unmap();
    writes.clear();
    // we get a implicit host -> device sync on the next queue submit.
}

void DescriptorBuffer::update(const CommandBufferHandle& cmd) {
    // for (uint32_t i = 0; i < writes.size(); i++) {
    //     const QueueElement& write = writes[i];
    //     apply_update_for(write.binding, write.array_element);
    //     cmd->update(buffer, write.offset, write.size, host_buffer.data() + write.offset);
    // }
}

} // namespace merian
