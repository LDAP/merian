#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

// Allocates one set for each layout
inline std::vector<vk::DescriptorSet>
allocate_descriptor_sets(vk::Device& device, vk::DescriptorPool pool, std::vector<vk::DescriptorSetLayout> layouts) {
    vk::DescriptorSetAllocateInfo info{pool, layouts};
    return device.allocateDescriptorSets(info);
}

// Allocates `count` sets for the supplied layout.
inline std::vector<vk::DescriptorSet>
allocate_descriptor_sets(vk::Device& device, vk::DescriptorPool pool, vk::DescriptorSetLayout layout, uint32_t count) {
    std::vector<vk::DescriptorSetLayout> layouts(count, layout);
    return allocate_descriptor_sets(device, pool, layouts);
}

// Updates the vk::DescriptorSet immediately (!) to point to the supplied buffer.
inline void update_descriptor_set_buffer(vk::Device& device,
                                         vk::DescriptorSet& set,
                                         uint32_t binding,
                                         vk::Buffer& buffer,
                                         vk::DeviceSize offset = 0,
                                         vk::DeviceSize range = VK_WHOLE_SIZE,
                                         vk::DescriptorType type = vk::DescriptorType::eStorageBuffer,
                                         uint32_t dst_array_element = 0,
                                         uint32_t descriptor_count = 1) {
    vk::DescriptorBufferInfo buffer_info{buffer, offset, range};
    vk::WriteDescriptorSet write{set, binding, dst_array_element, descriptor_count, type};
    device.updateDescriptorSets(1, &write, 0, nullptr);
}

} // namespace merian
