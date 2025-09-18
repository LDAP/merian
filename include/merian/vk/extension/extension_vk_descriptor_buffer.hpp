#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkDescriptorBuffer : public Extension {
  public:
    ExtensionVkDescriptorBuffer() : Extension("ExtensionVkDescriptorBuffer") {}
    ~ExtensionVkDescriptorBuffer() {}

    std::vector<const char*>
    required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const override {
        return {VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME};
    }

    void* pnext_get_properties_2(void* const p_next) override {
        descriptor_buffer_properties.pNext = p_next;
        return &descriptor_buffer_properties;
    }

    const vk::PhysicalDeviceDescriptorBufferPropertiesEXT&
    get_descriptor_buffer_properties() const {
        return descriptor_buffer_properties;
    }

    std::size_t descriptor_size_for_type(const vk::DescriptorType type) const {
        switch (type) {

        case vk::DescriptorType::eSampler:
            return descriptor_buffer_properties.samplerDescriptorSize;
        case vk::DescriptorType::eCombinedImageSampler:
            return descriptor_buffer_properties.combinedImageSamplerDescriptorSize;
        case vk::DescriptorType::eSampledImage:
            return descriptor_buffer_properties.sampledImageDescriptorSize;
        case vk::DescriptorType::eStorageImage:
            return descriptor_buffer_properties.storageImageDescriptorSize;
        case vk::DescriptorType::eUniformTexelBuffer:
            return descriptor_buffer_properties.uniformTexelBufferDescriptorSize;
        case vk::DescriptorType::eStorageTexelBuffer:
            return descriptor_buffer_properties.storageTexelBufferDescriptorSize;
        case vk::DescriptorType::eUniformBuffer:
            return descriptor_buffer_properties.uniformBufferDescriptorSize;
        case vk::DescriptorType::eStorageBuffer:
            return descriptor_buffer_properties.storageBufferDescriptorSize;
        case vk::DescriptorType::eInputAttachment:
            return descriptor_buffer_properties.inputAttachmentDescriptorSize;
        case vk::DescriptorType::eAccelerationStructureKHR:
        case vk::DescriptorType::eAccelerationStructureNV:
            return descriptor_buffer_properties.accelerationStructureDescriptorSize;
        case vk::DescriptorType::eUniformBufferDynamic:
        case vk::DescriptorType::eStorageBufferDynamic:
        case vk::DescriptorType::eInlineUniformBlock:
        case vk::DescriptorType::eSampleWeightImageQCOM:
        case vk::DescriptorType::eBlockMatchImageQCOM:
        case vk::DescriptorType::eMutableEXT:
        default:
            throw std::invalid_argument{
                fmt::format("{} not supported in descriptor buffers.", vk::to_string(type))};
        }
    }

  private:
    vk::PhysicalDeviceDescriptorBufferPropertiesEXT descriptor_buffer_properties;
};

} // namespace merian
