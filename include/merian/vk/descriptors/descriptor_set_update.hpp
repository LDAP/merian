#pragma once

#include "merian/vk/descriptors/descriptor_set.hpp"

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

// Utility class to update desriptors of a DescriptorSet.
//
// This can be used to `bind` buffers, images and acceleration structures to DescriptorSets.
// The binding type is automatically determined using the DescriptorSets and the binding index, you
// can use the *_type methods if you want to overwrite the type.
class DescriptorSetUpdate {

  public:
    DescriptorSetUpdate(const std::shared_ptr<DescriptorSet> set) : set(set) {
        assert(set);
    }

    // Bind `buffer` at the binding point `binding` of DescriptorSet `set`.
    // The type is automatically determined from the set using the binding index.
    DescriptorSetUpdate& write_descriptor_buffer(uint32_t binding,
                                                 vk::Buffer& buffer,
                                                 vk::DeviceSize offset = 0,
                                                 vk::DeviceSize range = VK_WHOLE_SIZE,
                                                 uint32_t dst_array_element = 0,
                                                 uint32_t descriptor_count = 1) {
        return write_descriptor_buffer_type(binding, buffer, set->get_type_for_binding(binding),
                                            offset, range, dst_array_element, descriptor_count);
    }

    // Bind `buffer` at the binding point `binding` of DescriptorSet `set`.
    // The type is automatically determined from the set using the binding index.
    DescriptorSetUpdate&
    write_descriptor_buffer_type(uint32_t binding,
                                 vk::Buffer& buffer,
                                 vk::DescriptorType type = vk::DescriptorType::eStorageBuffer,
                                 vk::DeviceSize offset = 0,
                                 vk::DeviceSize range = VK_WHOLE_SIZE,
                                 uint32_t dst_array_element = 0,
                                 uint32_t descriptor_count = 1) {
        write_buffer_infos.emplace_back(
            std::make_unique<vk::DescriptorBufferInfo>(buffer, offset, range));
        vk::WriteDescriptorSet write{*set,
                                     binding,
                                     dst_array_element,
                                     descriptor_count,
                                     type,
                                     {},
                                     write_buffer_infos.back().get()};
        writes.push_back(write);
        return *this;
    }

    // Bind `acceleration_structure` at the binding point `binding` of DescriptorSet `set`.
    DescriptorSetUpdate& write_descriptor_acceleration_structure(
        uint32_t binding,
        const std::vector<vk::AccelerationStructureKHR>& acceleration_structures,
        uint32_t dst_array_element = 0,
        uint32_t descriptor_count = 1) {
        write_acceleration_structures.emplace_back(
            std::make_unique<vk::WriteDescriptorSetAccelerationStructureKHR>(
                acceleration_structures));
        vk::WriteDescriptorSet write{*set,
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
    // The type is automatically determined from the set using the binding index.
    DescriptorSetUpdate& write_descriptor_image(uint32_t binding,
                                                vk::Sampler& sampler,
                                                vk::ImageView& image_view,
                                                vk::ImageLayout& image_layout,
                                                uint32_t dst_array_element = 0,
                                                uint32_t descriptor_count = 1) {
        return write_descriptor_image_type(binding, set->get_type_for_binding(binding), sampler,
                                           image_view, image_layout, dst_array_element,
                                           descriptor_count);
    }

    // Bind `sampler` at the binding point `binding` of DescriptorSet `set`.
    DescriptorSetUpdate& write_descriptor_image_type(uint32_t binding,
                                                     vk::DescriptorType type,
                                                     vk::Sampler& sampler,
                                                     vk::ImageView& view,
                                                     vk::ImageLayout& image_layout,
                                                     uint32_t dst_array_element = 0,
                                                     uint32_t descriptor_count = 1) {
        write_image_infos.emplace_back(
            std::make_unique<vk::DescriptorImageInfo>(sampler, view, image_layout));
        vk::WriteDescriptorSet write{*set,
                                     binding,
                                     dst_array_element,
                                     descriptor_count,
                                     type,
                                     write_image_infos.back().get()};

        return *this;
    }

    // Updates the vk::DescriptorSet immediately (!) to point to the configured resources.
    void update(SharedContext context) {
        assert(writes.size() == write_buffer_infos.size() + write_image_infos.size() +
                                    write_acceleration_structures.size());
        context->device.updateDescriptorSets(writes, {});
    }

    // Start a new update. If set == nullptr then the current set is reused.
    void next(const std::shared_ptr<DescriptorSet> set = nullptr) {
        if (set)
            this->set = set;

        writes.clear();
        write_buffer_infos.clear();
        write_image_infos.clear();
        write_acceleration_structures.clear();
    }

  private:
    std::shared_ptr<DescriptorSet> set;

    std::vector<vk::WriteDescriptorSet> writes;

    // vk::WriteDescriptorSet takes pointers. We must ensure that these stay valid until update() ->
    // use unique ptrs
    std::vector<std::unique_ptr<vk::DescriptorBufferInfo>> write_buffer_infos;
    std::vector<std::unique_ptr<vk::DescriptorImageInfo>> write_image_infos;
    std::vector<std::unique_ptr<vk::WriteDescriptorSetAccelerationStructureKHR>>
        write_acceleration_structures;
};

} // namespace merian
