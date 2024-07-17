#pragma once

#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <memory>
#include <optional>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

// Utility class to update desriptors of a DescriptorSet.
//
// This can be used to `bind` buffers, images and acceleration structures to DescriptorSets.
// The binding type is automatically determined using the DescriptorSets and the binding index.
// However, you can use the *_type methods if you want to overwrite the type.
//
// From the spec: The operations described by pDescriptorWrites are performed first, followed by the
// operations described by pDescriptorCopies. Within each array, the operations are performed in the
// order they appear in the array.
class DescriptorSetUpdate {

  public:
    DescriptorSetUpdate(const std::shared_ptr<DescriptorSet> set) : set(set) {
        assert(set);
    }

    // Bind `buffer` at the binding point `binding` of DescriptorSet `set`.
    // The type is automatically determined from the set using the binding index.
    DescriptorSetUpdate& write_descriptor_buffer(const uint32_t binding,
                                                 const BufferHandle& buffer,
                                                 const vk::DeviceSize offset = 0,
                                                 const vk::DeviceSize range = VK_WHOLE_SIZE,
                                                 const uint32_t dst_array_element = 0,
                                                 const uint32_t descriptor_count = 1) {
        return write_descriptor_buffer_type(binding, *buffer, set->get_type_for_binding(binding),
                                            offset, range, dst_array_element, descriptor_count);
    }

    // Bind `buffer` at the binding point `binding` of DescriptorSet `set`.
    // The type is automatically determined from the set using the binding index.
    DescriptorSetUpdate& write_descriptor_buffer(const uint32_t binding,
                                                 const vk::Buffer& buffer,
                                                 const vk::DeviceSize offset = 0,
                                                 const vk::DeviceSize range = VK_WHOLE_SIZE,
                                                 const uint32_t dst_array_element = 0,
                                                 const uint32_t descriptor_count = 1) {
        return write_descriptor_buffer_type(binding, buffer, set->get_type_for_binding(binding),
                                            offset, range, dst_array_element, descriptor_count);
    }

    // Bind `buffer` at the binding point `binding` of DescriptorSet `set`.
    // The type is automatically determined from the set using the binding index.
    DescriptorSetUpdate&
    write_descriptor_buffer_type(const uint32_t binding,
                                 const vk::Buffer& buffer,
                                 const vk::DescriptorType type = vk::DescriptorType::eStorageBuffer,
                                 const vk::DeviceSize offset = 0,
                                 const vk::DeviceSize range = VK_WHOLE_SIZE,
                                 const uint32_t dst_array_element = 0,
                                 const uint32_t descriptor_count = 1) {
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
        const uint32_t binding,
        const vk::AccelerationStructureKHR& acceleration_structure,
        const uint32_t dst_array_element = 0,
        const uint32_t descriptor_count = 1) {
        write_acceleration_structures.emplace_back(
            std::make_unique<vk::WriteDescriptorSetAccelerationStructureKHR>(
                1, &acceleration_structure));
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

    // The type is automatically determined from the set using the binding index.
    // With access_layout you can overwrite the layout that the image has "when it is accessed using
    // the descriptor". If std::nullopt the current layout is used.
    DescriptorSetUpdate&
    write_descriptor_texture(const uint32_t binding,
                             const TextureHandle texture,
                             const uint32_t dst_array_element = 0,
                             const uint32_t descriptor_count = 1,
                             const std::optional<vk::ImageLayout> access_layout = std::nullopt) {
        return write_descriptor_image_type(
            binding, set->get_type_for_binding(binding), texture->get_view(),
            access_layout.value_or(texture->get_current_layout()), *texture->get_sampler(),
            dst_array_element, descriptor_count);
    }

    // Bind `sampler` at the binding point `binding` of DescriptorSet `set`.
    // The type is automatically determined from the set using the binding index.
    DescriptorSetUpdate&
    write_descriptor_image(const uint32_t binding,
                           const vk::ImageView& image_view,
                           const vk::ImageLayout& image_layout = vk::ImageLayout::eGeneral,
                           const vk::Sampler& sampler = {},
                           const uint32_t dst_array_element = 0,
                           const uint32_t descriptor_count = 1) {
        return write_descriptor_image_type(binding, set->get_type_for_binding(binding), image_view,
                                           image_layout, sampler, dst_array_element,
                                           descriptor_count);
    }

    // Bind `sampler` at the binding point `binding` of DescriptorSet `set`.
    DescriptorSetUpdate&
    write_descriptor_image_type(const uint32_t binding,
                                const vk::DescriptorType type,
                                const vk::ImageView& view,
                                const vk::ImageLayout& image_layout = vk::ImageLayout::eGeneral,
                                const vk::Sampler& sampler = {},
                                const uint32_t dst_array_element = 0,
                                const uint32_t descriptor_count = 1) {
        write_image_infos.emplace_back(
            std::make_unique<vk::DescriptorImageInfo>(sampler, view, image_layout));
        vk::WriteDescriptorSet write{*set,
                                     binding,
                                     dst_array_element,
                                     descriptor_count,
                                     type,
                                     write_image_infos.back().get()};
        writes.push_back(write);

        return *this;
    }

    uint32_t count() const noexcept {
        return writes.size();
    }

    bool empty() const noexcept {
        return writes.empty();
    }

    // Updates the vk::DescriptorSet immediately (!) to point to the configured resources.
    void update(ContextHandle context) {
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
