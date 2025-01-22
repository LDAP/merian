#pragma once

#include "merian/vk/descriptors/descriptor_pool.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <memory>
#include <variant>
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
    DescriptorSet(const DescriptorPoolHandle& pool)
        : pool(pool), layout(pool->get_layout()),
          resource_index_for_binding(layout->get_bindings().size(), 0) {
        SPDLOG_DEBUG("allocating DescriptorSet ({})", fmt::ptr(this));
        set = allocate_descriptor_set(*pool->get_context(), *pool, *pool->get_layout());

        const auto& bindings = layout->get_bindings();
        if (!bindings.empty()) {
            for (uint32_t i = 1; i < bindings.size(); i++) {
                resource_index_for_binding[i] =
                    bindings[i - 1].descriptorCount + resource_index_for_binding[i - 1];
            }
            resources.resize(resource_index_for_binding.back() + bindings.back().descriptorCount);
        }
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
                                                 const uint32_t dst_array_element = 0) {
        write_resources.emplace_back(buffer);
        write_infos.emplace_back(vk::DescriptorBufferInfo(*buffer, offset, range));
        writes.emplace_back(vk::WriteDescriptorSet{
            set,
            binding,
            dst_array_element,
            1,
            layout->get_type_for_binding(binding),
        });
        return *this;
    }

    // The type is automatically determined from the set using the binding index.
    // With access_layout you can overwrite the layout that the image has "when it is accessed
    // using the descriptor". If std::nullopt the current layout is used.
    DescriptorSet& queue_descriptor_write_texture(
        const uint32_t binding,
        const TextureHandle& texture,
        const uint32_t dst_array_element = 0,
        const std::optional<vk::ImageLayout> access_layout = std::nullopt) {
        write_resources.emplace_back(texture);
        write_infos.emplace_back(
            vk::DescriptorImageInfo(*texture->get_sampler(), *texture->get_view(),
                                    access_layout.value_or(texture->get_current_layout())));
        writes.emplace_back(vk::WriteDescriptorSet{
            set,
            binding,
            dst_array_element,
            1,
            layout->get_type_for_binding(binding),
        });

        return *this;
    }

    // Bind `acceleration_structure` at the binding point `binding` of DescriptorSet `set`.
    DescriptorSet& queue_descriptor_write_acceleration_structure(
        const uint32_t binding,
        const AccelerationStructureHandle& acceleration_structure,
        const uint32_t dst_array_element = 0) {
        write_resources.emplace_back(acceleration_structure);
        write_infos.emplace_back(
            vk::WriteDescriptorSetAccelerationStructureKHR(1, *acceleration_structure));
        writes.emplace_back(vk::WriteDescriptorSet{
            set,
            binding,
            dst_array_element,
            1,
            vk::DescriptorType::eAccelerationStructureKHR,
        });

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
    // If you get validation errors or crashes following a call to this method, you likely tried
    // to update a DescriptorSet that is currently used in an pending or executing command
    // buffer.
    void update() {
        // for now descriptor count is always 1.
        assert(writes.size() == write_resources.size());
        assert(writes.size() == write_infos.size());

        if (!has_updates()) {
            return;
        }

        for (uint32_t i = 0; i < writes.size(); i++) {
            vk::WriteDescriptorSet& write = writes[i];

            if (std::holds_alternative<vk::DescriptorBufferInfo>(write_infos[i])) {
                write.pBufferInfo = &std::get<vk::DescriptorBufferInfo>(write_infos[i]);
            } else if (std::holds_alternative<vk::DescriptorImageInfo>(write_infos[i])) {
                write.pImageInfo = &std::get<vk::DescriptorImageInfo>(write_infos[i]);
            } else if (std::holds_alternative<vk::WriteDescriptorSetAccelerationStructureKHR>(
                           write_infos[i])) {
                write.pNext =
                    &std::get<vk::WriteDescriptorSetAccelerationStructureKHR>(write_infos[i]);
            } else {
                assert(false);
            }

            // For now only descriptorCount 1 is implemented. Otherwise: If the dstBinding has fewer
            // than descriptorCount array elements remaining starting from dstArrayElement, then the
            // remainder will be used to update the subsequent binding - dstBinding+1 starting at
            // array element zero.
            assert(write.descriptorCount == 1);
            resources[resource_index_for_binding[write.dstBinding] + write.dstArrayElement] =
                write_resources[i];
        }

        pool->get_context()->device.updateDescriptorSets(writes, {});

        writes.clear();
        write_infos.clear();
        write_resources.clear();
    }

  private:
    const std::shared_ptr<DescriptorPool> pool;
    const std::shared_ptr<DescriptorSetLayout> layout;
    vk::DescriptorSet set;

    // Has entry for each array element. Use resource_index_for_binding to access.
    std::vector<ResourceHandle> resources;
    std::vector<uint32_t> resource_index_for_binding;

    // ---------------------------------------------------------------------
    // Queued Updates

    // all with the same length.
    // Pointers are set in update() depending on the descriptor type.
    std::vector<vk::WriteDescriptorSet> writes;
    std::vector<ResourceHandle> write_resources;
    std::vector<std::variant<vk::DescriptorBufferInfo,
                             vk::DescriptorImageInfo,
                             vk::WriteDescriptorSetAccelerationStructureKHR>>
        write_infos;
};

using DescriptorSetHandle = std::shared_ptr<DescriptorSet>;

} // namespace merian
