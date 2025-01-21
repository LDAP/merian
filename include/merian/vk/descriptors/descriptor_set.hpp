#pragma once

#include "merian/vk/descriptors/descriptor_pool.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

// Allocates one set for each layout
inline std::vector<vk::DescriptorSet>
allocate_descriptor_sets(const vk::Device& device,
                         const vk::DescriptorPool& pool,
                         const std::vector<vk::DescriptorSetLayout>& layouts) {
    vk::DescriptorSetAllocateInfo info{pool, layouts};
    return device.allocateDescriptorSets(info);
}

// Allocates `count` sets for the supplied layout.
inline std::vector<vk::DescriptorSet>
allocate_descriptor_sets(const vk::Device& device,
                         const vk::DescriptorPool& pool,
                         const vk::DescriptorSetLayout& layout,
                         uint32_t count) {
    std::vector<vk::DescriptorSetLayout> layouts(count, layout);
    return allocate_descriptor_sets(device, pool, layouts);
}

inline vk::DescriptorSet allocate_descriptor_set(const vk::Device& device,
                                                 const vk::DescriptorPool& pool,
                                                 const vk::DescriptorSetLayout& layout) {
    return allocate_descriptor_sets(device, pool, {layout})[0];
}

// A DescriptorSet that knows its layout -> Can be used to simplify DescriptorSet updates.
// DescriptorsSet updates are queued until they are executed with a call to update(). In this case
// the update is performed immediately.
// The DescriptorSet holds references to the resources that are bound to it.
class DescriptorSet : public std::enable_shared_from_this<DescriptorSet>, public Object {

  public:
    // Allocates a DescriptorSet that matches the layout that is attached to the Pool
    DescriptorSet(const DescriptorPoolHandle& pool) : pool(pool), layout(pool->get_layout()) {
        SPDLOG_DEBUG("allocating DescriptorSet ({})", fmt::ptr(this));
        set = allocate_descriptor_set(*pool->get_context(), *pool, *pool->get_layout());
    }

    ~DescriptorSet() {
        if (pool->get_create_flags() & vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet) {
            // DescriptorSet can be given back to the DescriptorPool
            SPDLOG_DEBUG("freeing DescriptorSet ({})", fmt::ptr(this));
            pool->get_context()->device.freeDescriptorSets(*pool, set);
        } else {
            SPDLOG_DEBUG("destroying DescriptorSet ({}) but not freeing since the pool was not "
                         "created with the {} bit set.",
                         fmt::ptr(this),
                         vk::to_string(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet));
        }
    }

    operator const vk::DescriptorSet&() const {
        return set;
    }

    operator const vk::DescriptorSetLayout&() const {
        return *layout;
    }

    const vk::DescriptorSet& operator*() const {
        return set;
    }

    const vk::DescriptorSet& get_descriptor_set() const {
        return set;
    }

    const DescriptorSetLayoutHandle& get_layout() const {
        return layout;
    }

    const vk::DescriptorType& get_type_for_binding(uint32_t binding) const {
        return layout->get_bindings()[binding].descriptorType;
    }

    // ---------------------------------------------------------------------
    // Updates

    // Bind `buffer` at the binding point `binding` of DescriptorSet `set`.
    // The type is automatically determined from the set using the binding index.
    DescriptorSet& queue_descriptor_write_buffer(const uint32_t binding,
                                                 const BufferHandle& buffer,
                                                 const vk::DeviceSize offset = 0,
                                                 const vk::DeviceSize range = VK_WHOLE_SIZE,
                                                 const uint32_t dst_array_element = 0,
                                                 const uint32_t descriptor_count = 1) {
        resources.emplace_back(buffer);
        return write_descriptor_buffer_type(binding, *buffer, get_type_for_binding(binding), offset,
                                            range, dst_array_element, descriptor_count);
    }

    // The type is automatically determined from the set using the binding index.
    // With access_layout you can overwrite the layout that the image has "when it is accessed using
    // the descriptor". If std::nullopt the current layout is used.
    DescriptorSet& queue_descriptor_write_texture(
        const uint32_t binding,
        const TextureHandle& texture,
        const uint32_t dst_array_element = 0,
        const uint32_t descriptor_count = 1,
        const std::optional<vk::ImageLayout> access_layout = std::nullopt) {
        resources.emplace_back(texture);
        return write_descriptor_image_type(
            binding, get_type_for_binding(binding), *texture->get_view(),
            access_layout.value_or(texture->get_current_layout()), *texture->get_sampler(),
            dst_array_element, descriptor_count);
    }

    // Bind `acceleration_structure` at the binding point `binding` of DescriptorSet `set`.
    DescriptorSet& queue_descriptor_write_acceleration_structure(
        const uint32_t binding,
        const vk::AccelerationStructureKHR& acceleration_structure,
        const uint32_t dst_array_element = 0,
        const uint32_t descriptor_count = 1) {
        write_acceleration_structures.emplace_back(
            std::make_unique<vk::WriteDescriptorSetAccelerationStructureKHR>(
                1, &acceleration_structure));
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

    // Returns the number of queued updates which will be executed with a call to update().
    uint32_t update_count() const noexcept {
        return writes.size();
    }

    // Returns weather there are updates queued for this descriptor set.
    bool has_updates() const noexcept {
        return !writes.empty();
    }

    // Updates the vk::DescriptorSet immediately (!) to point to the configured resources.
    //
    // If you get validation errors or crashes following a call to this method, you likely tried to
    // update a DescriptorSet that is currently used in an pending or executing command buffer.
    void update() {
        if (!has_updates()) {
            return;
        }

        assert(writes.size() == write_buffer_infos.size() + write_image_infos.size() +
                                    write_acceleration_structures.size());
        pool->get_context()->device.updateDescriptorSets(writes, {});

        writes.clear();
        write_buffer_infos.clear();
        write_image_infos.clear();
        write_acceleration_structures.clear();
        resources.clear();
    }

  private:
    // Bind `buffer` at the binding point `binding`.
    // The type is automatically determined from the set using the binding index.
    DescriptorSet& write_descriptor_buffer(const uint32_t binding,
                                           const vk::Buffer& buffer,
                                           const vk::DeviceSize offset = 0,
                                           const vk::DeviceSize range = VK_WHOLE_SIZE,
                                           const uint32_t dst_array_element = 0,
                                           const uint32_t descriptor_count = 1) {
        return write_descriptor_buffer_type(binding, buffer, get_type_for_binding(binding), offset,
                                            range, dst_array_element, descriptor_count);
    }

    // Bind `sampler` at the binding point `binding``.
    // The type is automatically determined from the set using the binding index.
    DescriptorSet&
    write_descriptor_image(const uint32_t binding,
                           const vk::ImageView& image_view,
                           const vk::ImageLayout& image_layout = vk::ImageLayout::eGeneral,
                           const vk::Sampler& sampler = {},
                           const uint32_t dst_array_element = 0,
                           const uint32_t descriptor_count = 1) {
        return write_descriptor_image_type(binding, get_type_for_binding(binding), image_view,
                                           image_layout, sampler, dst_array_element,
                                           descriptor_count);
    }

    // Bind `sampler` at the binding point `binding`.
    DescriptorSet&
    write_descriptor_image_type(const uint32_t binding,
                                const vk::DescriptorType type,
                                const vk::ImageView& view,
                                const vk::ImageLayout& image_layout = vk::ImageLayout::eGeneral,
                                const vk::Sampler& sampler = {},
                                const uint32_t dst_array_element = 0,
                                const uint32_t descriptor_count = 1) {
        write_image_infos.emplace_back(
            std::make_unique<vk::DescriptorImageInfo>(sampler, view, image_layout));
        vk::WriteDescriptorSet write{set,
                                     binding,
                                     dst_array_element,
                                     descriptor_count,
                                     type,
                                     write_image_infos.back().get()};
        writes.push_back(write);

        return *this;
    }

    // Bind `buffer` at the binding point `binding` of DescriptorSet `set`.
    // The type is automatically determined from the set using the binding index.
    DescriptorSet&
    write_descriptor_buffer_type(const uint32_t binding,
                                 const vk::Buffer& buffer,
                                 const vk::DescriptorType type = vk::DescriptorType::eStorageBuffer,
                                 const vk::DeviceSize offset = 0,
                                 const vk::DeviceSize range = VK_WHOLE_SIZE,
                                 const uint32_t dst_array_element = 0,
                                 const uint32_t descriptor_count = 1) {
        write_buffer_infos.emplace_back(
            std::make_unique<vk::DescriptorBufferInfo>(buffer, offset, range));
        vk::WriteDescriptorSet write{set,
                                     binding,
                                     dst_array_element,
                                     descriptor_count,
                                     type,
                                     {},
                                     write_buffer_infos.back().get()};
        writes.push_back(write);
        return *this;
    }

  private:
    const std::shared_ptr<DescriptorPool> pool;
    const std::shared_ptr<DescriptorSetLayout> layout;
    vk::DescriptorSet set;

    std::vector<ResourceHandle> bound_resources;

    // ---------------------------------------------------------------------
    // Queued Updates

    std::vector<vk::WriteDescriptorSet> writes;
    std::vector<ResourceHandle> resources;

    // vk::WriteDescriptorSet takes pointers. We must ensure that these stay valid until update() ->
    // use unique ptrs
    std::vector<std::unique_ptr<vk::DescriptorBufferInfo>> write_buffer_infos;
    std::vector<std::unique_ptr<vk::DescriptorImageInfo>> write_image_infos;
    std::vector<std::unique_ptr<vk::WriteDescriptorSetAccelerationStructureKHR>>
        write_acceleration_structures;
};

using DescriptorSetHandle = std::shared_ptr<DescriptorSet>;

} // namespace merian
