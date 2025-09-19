#pragma once

#include "merian/utils/pointer.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <memory>
#include <variant>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

// A base class for container that hold descriptors (sets, buffers)
class DescriptorContainer : public std::enable_shared_from_this<DescriptorContainer>,
                            public Object {

  protected:
    using DescriptorInfo = std::variant<vk::DescriptorBufferInfo,
                                        vk::DescriptorImageInfo,
                                        vk::WriteDescriptorSetAccelerationStructureKHR>;

  public:
    static const uint32_t NO_DESCRIPTOR_BINDING = -1u;

    DescriptorContainer(const DescriptorSetLayoutHandle& layout)
        : layout(layout), resource_index_for_binding(layout->get_bindings().size(), 0) {

        const auto& bindings = layout->get_bindings();
        if (!bindings.empty()) {
            for (uint32_t i = 1; i < bindings.size(); i++) {
                resource_index_for_binding[i] =
                    bindings[i - 1].descriptorCount + resource_index_for_binding[i - 1];
            }
            descriptor_count = resource_index_for_binding.back() + bindings.back().descriptorCount;

            resources.resize(descriptor_count);
            write_resources.resize(descriptor_count);
            write_infos.resize(descriptor_count);
        }
    }

    virtual ~DescriptorContainer();

    // ----------------------------------------------------

    const DescriptorSetLayoutHandle& get_layout() const {
        return layout;
    }

    BufferHandle get_buffer_at(const uint32_t binding, const uint32_t array_element = 0) const {
        const std::shared_ptr<Resource>& resource =
            get_bindable_resource_at(binding, array_element);

        if (BufferHandle buffer = std::dynamic_pointer_cast<Buffer>(resource); buffer) {
            return buffer;
        }

        throw std::runtime_error{
            fmt::format("no buffer at binding {} (array element {})", binding, array_element)};
    }

    ImageViewHandle get_view_at(const uint32_t binding, const uint32_t array_element = 0) const {
        const std::shared_ptr<Resource>& resource =
            get_bindable_resource_at(binding, array_element);

        if (ImageViewHandle image_view = std::dynamic_pointer_cast<ImageView>(resource);
            image_view) {
            return image_view;
        }
        if (TextureHandle texture = std::dynamic_pointer_cast<Texture>(resource); texture) {
            return texture->get_view();
        }

        throw std::runtime_error{
            fmt::format("no view at binding {} (array element {})", binding, array_element)};
    }

    TextureHandle get_texture_at(const uint32_t binding, const uint32_t array_element = 0) const {
        const std::shared_ptr<Resource>& resource =
            get_bindable_resource_at(binding, array_element);
        if (TextureHandle texture = std::dynamic_pointer_cast<Texture>(resource); texture) {
            return texture;
        }

        throw std::runtime_error{
            fmt::format("no texture at binding {} (array element {})", binding, array_element)};
    }

    AccelerationStructureHandle
    get_acceleration_structure_at(const uint32_t binding, const uint32_t array_element = 0) const {
        const std::shared_ptr<Resource>& resource =
            get_bindable_resource_at(binding, array_element);
        if (AccelerationStructureHandle as =
                std::dynamic_pointer_cast<AccelerationStructure>(resource);
            as) {
            return as;
        }

        throw std::runtime_error{fmt::format(
            "no acceleration_structure at binding {} (array element {})", binding, array_element)};
    }

    // --------------------------------------

    DescriptorContainer& queue_descriptor_write_buffer(const uint32_t binding,
                                                       const BufferHandle& buffer,
                                                       const vk::DeviceSize offset = 0,
                                                       const vk::DeviceSize range = VK_WHOLE_SIZE,
                                                       const uint32_t dst_array_element = 0) {
        assert(buffer);

        const uint32_t index = resource_index_for_binding[binding] + dst_array_element;
        write_infos[index] = buffer->get_descriptor_info(offset, range);
        if (!write_resources[index]) {
            // minimize writes (and allow move in apply_update_for)
            queue_write(vk::WriteDescriptorSet{
                VK_NULL_HANDLE,
                binding,
                dst_array_element,
                1,
                get_layout()->get_type_for_binding(binding),
                VK_NULL_HANDLE,
                &std::get<vk::DescriptorBufferInfo>(write_infos[index]),
            });
        }
        write_resources[index] = buffer;
        return *this;
    }

    DescriptorContainer& queue_descriptor_write_image(
        const uint32_t binding,
        const ImageViewHandle& image_view,
        const uint32_t dst_array_element = 0,
        const std::optional<vk::ImageLayout> access_layout = std::nullopt) {
        assert(image_view);

        const uint32_t index = resource_index_for_binding[binding] + dst_array_element;
        write_infos[index] = image_view->get_descriptor_info(access_layout);
        if (!write_resources[index]) {
            // minimize writes (and allow move in apply_update_for)
            queue_write(vk::WriteDescriptorSet{
                VK_NULL_HANDLE,
                binding,
                dst_array_element,
                1,
                get_layout()->get_type_for_binding(binding),
                &std::get<vk::DescriptorImageInfo>(write_infos[index]),
            });
        }
        write_resources[index] = image_view;

        return *this;
    }

    DescriptorContainer& queue_descriptor_write_texture(
        const uint32_t binding,
        const TextureHandle& texture,
        const uint32_t dst_array_element = 0,
        const std::optional<vk::ImageLayout> access_layout = std::nullopt) {
        assert(texture);

        const uint32_t index = resource_index_for_binding[binding] + dst_array_element;
        write_infos[index] = texture->get_descriptor_info(access_layout);
        if (!write_resources[index]) {
            // minimize writes (and allow move in apply_update_for)
            queue_write(vk::WriteDescriptorSet{
                VK_NULL_HANDLE,
                binding,
                dst_array_element,
                1,
                get_layout()->get_type_for_binding(binding),
                &std::get<vk::DescriptorImageInfo>(write_infos[index]),
            });
        }
        write_resources[index] = texture;

        return *this;
    }

    DescriptorContainer& queue_descriptor_write_texture(
        const uint32_t binding,
        const ImageViewHandle& view,
        const SamplerHandle& sampler,
        const uint32_t dst_array_element = 0,
        const std::optional<vk::ImageLayout> access_layout = std::nullopt) {
        assert(view);
        assert(sampler);

        return queue_descriptor_write_texture(binding, Texture::create(view, sampler),
                                              dst_array_element, access_layout);
    }

    // Bind `acceleration_structure` at the binding point `binding` of DescriptorContainer `set`.
    DescriptorContainer& queue_descriptor_write_acceleration_structure(
        const uint32_t binding,
        const AccelerationStructureHandle& acceleration_structure,
        const uint32_t dst_array_element = 0) {
        assert(acceleration_structure);

        const uint32_t index = resource_index_for_binding[binding] + dst_array_element;
        write_infos[index] = acceleration_structure->get_descriptor_info();
        if (!write_resources[index]) {
            // minimize writes (and allow move in apply_update_for)
            queue_write(vk::WriteDescriptorSet{
                VK_NULL_HANDLE,
                binding,
                dst_array_element,
                1,
                vk::DescriptorType::eAccelerationStructureKHR,
                VK_NULL_HANDLE,
                VK_NULL_HANDLE,
                VK_NULL_HANDLE,
                &std::get<vk::WriteDescriptorSetAccelerationStructureKHR>(write_infos[index]),
            });
        }
        write_resources[index] = acceleration_structure;

        return *this;
    }

    // --------------------------------------

    uint32_t get_descriptor_count() const noexcept {
        return descriptor_count;
    }

    virtual uint32_t update_count() const noexcept = 0;

    virtual bool has_updates() const noexcept = 0;

    virtual void update() {
        throw std::runtime_error{"update on the CPU timeline not supported."};
    }

    virtual void update(const CommandBufferHandle& /*cmd*/) {
        throw std::runtime_error{"update on the GPU timeline not supported."};
    }

  protected:
    virtual void queue_write(vk::WriteDescriptorSet&& write) = 0;

    const std::shared_ptr<Resource>&
    get_bindable_resource_at(const uint32_t binding, const uint32_t array_element = 0) const {
        assert(binding < resource_index_for_binding.size());
        assert(array_element < layout->get_bindings()[binding].descriptorCount);

        return resources[resource_index_for_binding[binding] + array_element];
    }

    void apply_update_for(const uint32_t binding, const uint32_t array_element = 0) {
        assert(binding < resource_index_for_binding.size());
        assert(array_element < layout->get_bindings()[binding].descriptorCount);

        const uint32_t index = resource_index_for_binding[binding] + array_element;
        assert(index < resources.size());
        assert(index < write_resources.size());
        assert(write_resources[index]);

        resources[index] = std::move(write_resources[index]);
    }

  private:
    const DescriptorSetLayoutHandle layout;
    uint32_t descriptor_count = 0;

    std::vector<uint32_t> resource_index_for_binding;

    // Has entry for each array element. Use resource_index_for_binding to access.
    std::vector<std::shared_ptr<Resource>> resources;
    // Has entry for each array element. Use resource_index_for_binding to access.
    std::vector<std::shared_ptr<Resource>> write_resources;
    // Has entry for each array element. Use resource_index_for_binding to access.
    std::vector<std::variant<vk::DescriptorBufferInfo,
                             vk::DescriptorImageInfo,
                             vk::WriteDescriptorSetAccelerationStructureKHR>>
        write_infos;
};

using DescriptorContainerHandle = std::shared_ptr<DescriptorContainer>;

} // namespace merian
