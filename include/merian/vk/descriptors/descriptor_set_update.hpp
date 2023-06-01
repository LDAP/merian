#pragma once

#include "merian/vk/descriptors/descriptor_set.hpp"

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

// Utility class to update desriptors of a DescriptorSet.
// This can be used to `bind` buffers, images and acceleration structures to DescriptorSets.
class DescriptorSetUpdate {

  public:
    DescriptorSetUpdate() {}

    // Bind `buffer` at the binding point `binding` of DescriptorSet `set`. 
    DescriptorSetUpdate& write_descriptor_buffer(const DescriptorSet& set,
                                                 uint32_t binding,
                                                 vk::Buffer& buffer,
                                                 vk::DeviceSize offset = 0,
                                                 vk::DeviceSize range = VK_WHOLE_SIZE,
                                                 uint32_t dst_array_element = 0,
                                                 uint32_t descriptor_count = 1) {
        return write_descriptor_buffer(set, binding, buffer, offset, range, set.get_type_for_binding(binding),
                                       dst_array_element, descriptor_count);
    }

    // Bind `buffer` at the binding point `binding` of DescriptorSet `set`.
    DescriptorSetUpdate& write_descriptor_buffer(const vk::DescriptorSet& set,
                                                 uint32_t binding,
                                                 vk::Buffer& buffer,
                                                 vk::DeviceSize offset = 0,
                                                 vk::DeviceSize range = VK_WHOLE_SIZE,
                                                 vk::DescriptorType type = vk::DescriptorType::eStorageBuffer,
                                                 uint32_t dst_array_element = 0,
                                                 uint32_t descriptor_count = 1) {
        write_buffer_infos.emplace_back(std::make_unique<vk::DescriptorBufferInfo>(buffer, offset, range));
        vk::WriteDescriptorSet write{
            set, binding, dst_array_element, descriptor_count, type, {}, write_buffer_infos.back().get()};
        writes.push_back(write);
        return *this;
    }

    // Bind `acceleration_structure` at the binding point `binding` of DescriptorSet `set`.
    DescriptorSetUpdate&
    write_descriptor_acceleration_structure(const vk::DescriptorSet& set,
                                            uint32_t binding,
                                            std::vector<vk::AccelerationStructureKHR>& acceleration_structures,
                                            uint32_t dst_array_element = 0,
                                            uint32_t descriptor_count = 1) {
        write_acceleration_structures.emplace_back(
            std::make_unique<vk::WriteDescriptorSetAccelerationStructureKHR>(acceleration_structures));
        vk::WriteDescriptorSet write{set,
                                     binding,
                                     dst_array_element,
                                     descriptor_count,
                                     vk::DescriptorType::eAccelerationStructureKHR,
                                     {},
                                     {},
                                     {},
                                     write_acceleration_structures.back().get()};
        writes.push_back(write);
        return *this;
    }

    // Bind `sampler` at the binding point `binding` of DescriptorSet `set`.
    DescriptorSetUpdate& write_descriptor_image(const DescriptorSet& set,
                                                uint32_t binding,
                                                vk::Sampler& sampler,
                                                vk::ImageView& image_view,
                                                vk::ImageLayout& image_layout,
                                                uint32_t dst_array_element = 0,
                                                uint32_t descriptor_count = 1) {
        return write_descriptor_image(set, binding, set.get_type_for_binding(binding), sampler, image_view, image_layout,
                                      dst_array_element, descriptor_count);
    }

    // Bind `sampler` at the binding point `binding` of DescriptorSet `set`.
    DescriptorSetUpdate& write_descriptor_image(const vk::DescriptorSet& set,
                                                uint32_t binding,
                                                vk::DescriptorType type,
                                                vk::Sampler& sampler,
                                                vk::ImageView& view,
                                                vk::ImageLayout& image_layout,
                                                uint32_t dst_array_element = 0,
                                                uint32_t descriptor_count = 1) {
        write_image_infos.emplace_back(std::make_unique<vk::DescriptorImageInfo>(sampler, view, image_layout));
        vk::WriteDescriptorSet write{
            set, binding, dst_array_element, descriptor_count, type, write_image_infos.back().get()};

        return *this;
    }

    // Updates the vk::DescriptorSet immediately (!) to point to the configured resources.
    void update(vk::Device& device) {
        device.updateDescriptorSets(writes, {});
    }

  private:
    std::vector<vk::WriteDescriptorSet> writes;

    // vk::WriteDescriptorSet takes pointers. We must ensure that these stay valid until update() -> use unique ptrs
    std::vector<std::unique_ptr<vk::DescriptorBufferInfo>> write_buffer_infos;
    std::vector<std::unique_ptr<vk::DescriptorImageInfo>> write_image_infos;
    std::vector<std::unique_ptr<vk::WriteDescriptorSetAccelerationStructureKHR>> write_acceleration_structures;
};

} // namespace merian
