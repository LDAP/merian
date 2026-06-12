#include "merian/shader/shader_object.hpp"

#include <cassert>
#include <spdlog/spdlog.h>

namespace merian {

ShaderObject::ShaderObject(const ShaderObjectLayoutHandle& object_layout,
                           const ResourceAllocatorHandle& allocator)
    : object_layout(object_layout), allocator(allocator) {
    assert(object_layout);
    assert(allocator);

    if (object_layout->is_struct()) {
        slots.resize(object_layout->get_slot_count());
        subobjects.resize(object_layout->get_subobject_range_count());
        return;
    }

    const vk::DeviceSize uniform_size = object_layout->get_uniform_size();
    if (uniform_size > 0) {
        // dirty so the zero-initialized staging is uploaded even if never written
        uniform_staging.resize(uniform_size, 0);
        uniform_dirty = true;
        uniform_buffer =
            allocator->create_buffer(uniform_size, vk::BufferUsageFlagBits::eUniformBuffer |
                                                       vk::BufferUsageFlagBits::eTransferDst);
    }

    if (object_layout->is_parameter_block() && object_layout->has_bindings()) {
        descriptors = DescriptorStorage::create(object_layout->get_descriptor_set_layout());
        if (uniform_buffer) {
            descriptors->queue_descriptor_write_buffer(object_layout->get_uniform_buffer_binding(),
                                                       uniform_buffer, 0, uniform_size);
        }
    }
}

const ShaderObjectHandle& ShaderObject::get_element() {
    assert(object_layout->is_container());

    if (!element) {
        element = std::make_shared<ShaderObject>(object_layout->get_element_layout(), allocator);
        element->owner_container = weak_from_this();
        if (descriptors) {
            element->attach(shared_from_this(), object_layout->get_element_binding_offset());
        }
    }
    return element;
}

ShaderCursor ShaderObject::get_cursor() {
    if (object_layout->is_container()) {
        return get_element()->get_cursor();
    }
    return ShaderCursor(this);
}

// ---------------------------------------------------------------
// Sub-objects

const ShaderObjectHandle& ShaderObject::get_subobject(const uint32_t subobject_range_index) {
    if (object_layout->is_container()) {
        return get_element()->get_subobject(subobject_range_index);
    }
    assert(subobject_range_index < subobjects.size());
    return subobjects[subobject_range_index];
}

void ShaderObject::set_subobject(const uint32_t subobject_range_index,
                                 const ShaderObjectHandle& object) {
    if (object_layout->is_container()) {
        get_element()->set_subobject(subobject_range_index, object);
        return;
    }

    assert(subobject_range_index < subobjects.size());
    const auto& range = object_layout->get_subobject_range_info(subobject_range_index);
    assert(!object ||
           object->get_object_layout()->get_kind() == range.container_layout->get_kind());

    ShaderObjectHandle& current = subobjects[subobject_range_index];
    if (current == object) {
        return;
    }

    if (range.container_layout->is_constant_buffer()) {
        for_each_descriptor_target([&](ShaderObject& parameter_block, const uint32_t binding_base) {
            if (current) {
                current->get_element()->detach(
                    &parameter_block, binding_base + range.descriptor_slot_offset +
                                          range.container_layout->get_element_binding_offset());
            }
        });
    }

    current = object;
    if (!object || !range.container_layout->is_constant_buffer()) {
        // Nested ParameterBlocks own their descriptor set; nothing to write here.
        return;
    }

    for_each_descriptor_target([&](ShaderObject& parameter_block, const uint32_t binding_base) {
        attach_constant_buffer(object, range, parameter_block.shared_from_this(), binding_base);
    });
}

ShaderObjectHandle ShaderObject::create_subobject(const std::string& field_name) {
    if (object_layout->is_container()) {
        return get_element()->create_subobject(field_name);
    }

    auto* type_layout = object_layout->get_type_layout();
    const SlangInt field_index = type_layout->findFieldIndexByName(field_name.c_str());
    assert(field_index >= 0 && "Field not found");

    const uint32_t binding_range_index =
        type_layout->getFieldBindingRangeOffset(static_cast<uint32_t>(field_index));
    const int32_t subobject_range_index =
        object_layout->find_subobject_range_index(binding_range_index);
    assert(subobject_range_index >= 0 && "Field must be a ConstantBuffer or ParameterBlock");

    const auto& range_info = object_layout->get_subobject_range_info(subobject_range_index);
    assert(range_info.container_layout);

    return std::make_shared<ShaderObject>(range_info.container_layout, allocator);
}

void ShaderObject::set_subobject(const std::string& field_name, const ShaderObjectHandle& object) {
    if (object_layout->is_container()) {
        get_element()->set_subobject(field_name, object);
        return;
    }

    auto* type_layout = object_layout->get_type_layout();
    const SlangInt field_index = type_layout->findFieldIndexByName(field_name.c_str());
    assert(field_index >= 0 && "Field not found");

    const int32_t subobject_range_index = object_layout->find_subobject_range_index(
        type_layout->getFieldBindingRangeOffset(static_cast<uint32_t>(field_index)));
    assert(subobject_range_index >= 0 && "Field must be a ConstantBuffer or ParameterBlock");

    set_subobject(static_cast<uint32_t>(subobject_range_index), object);
}

// ---------------------------------------------------------------
// Attachment / replay

void ShaderObject::attach(const ShaderObjectHandle& parameter_block, const uint32_t binding_base) {
    assert(object_layout->is_struct());

    for (const auto& target : descriptor_targets) {
        if (target.parameter_block.lock() == parameter_block &&
            target.binding_base == binding_base) {
            return;
        }
    }

    descriptor_targets.push_back(DescriptorTarget{parameter_block, binding_base});
    replay_to(*parameter_block, binding_base);

    // Nested ConstantBuffers land in the same descriptor set.
    for (uint32_t subobject_range_index = 0; subobject_range_index < subobjects.size();
         subobject_range_index++) {
        const auto& subobject = subobjects[subobject_range_index];
        const auto& range = object_layout->get_subobject_range_info(subobject_range_index);
        if (subobject && range.container_layout->is_constant_buffer()) {
            attach_constant_buffer(subobject, range, parameter_block, binding_base);
        }
    }
}

void ShaderObject::detach(const ShaderObject* parameter_block, const uint32_t binding_base) {
    assert(object_layout->is_struct());

    for (auto it = descriptor_targets.begin(); it != descriptor_targets.end(); ++it) {
        if (it->parameter_block.lock().get() == parameter_block &&
            it->binding_base == binding_base) {
            descriptor_targets.erase(it);
            break;
        }
    }

    for (uint32_t subobject_range_index = 0; subobject_range_index < subobjects.size();
         subobject_range_index++) {
        const auto& subobject = subobjects[subobject_range_index];
        const auto& range = object_layout->get_subobject_range_info(subobject_range_index);
        if (subobject && range.container_layout->is_constant_buffer()) {
            subobject->get_element()->detach(
                parameter_block, binding_base + range.descriptor_slot_offset +
                                     range.container_layout->get_element_binding_offset());
        }
    }
}

void ShaderObject::attach_constant_buffer(const ShaderObjectHandle& constant_buffer,
                                          const ShaderObjectLayout::SubobjectRangeInfo& range,
                                          const ShaderObjectHandle& parameter_block,
                                          const uint32_t binding_base) {
    const auto& container_layout = range.container_layout;
    const uint32_t subobject_base = binding_base + range.descriptor_slot_offset;

    if (constant_buffer->uniform_buffer) {
        const uint32_t uniform_buffer_binding =
            subobject_base + container_layout->get_uniform_buffer_binding();
        parameter_block->for_each_descriptor_container([&](DescriptorContainer& container) {
            container.queue_descriptor_write_buffer(
                uniform_buffer_binding, constant_buffer->uniform_buffer, 0, VK_WHOLE_SIZE);
        });
    }

    constant_buffer->get_element()->attach(
        parameter_block, subobject_base + container_layout->get_element_binding_offset());
}

void ShaderObject::replay_to(ShaderObject& parameter_block, const uint32_t binding_base) {
    const uint32_t binding_range_count =
        static_cast<uint32_t>(object_layout->get_type_layout()->getBindingRangeCount());
    for (uint32_t binding_range_index = 0; binding_range_index < binding_range_count;
         binding_range_index++) {
        const auto& info = object_layout->get_binding_range_info(binding_range_index);
        for (uint32_t array_index = 0; array_index < info.count; array_index++) {
            const ResourceSlot& slot = slots[info.slot_base + array_index];
            if (!std::holds_alternative<std::monostate>(slot)) {
                apply_slot(slot, parameter_block, binding_base + info.binding, array_index);
            }
        }
    }
}

void ShaderObject::apply_slot(const ResourceSlot& slot,
                              ShaderObject& parameter_block,
                              const uint32_t binding,
                              const uint32_t array_index) {
    parameter_block.for_each_descriptor_container([&](DescriptorContainer& container) {
        if (const auto* image = std::get_if<ImageSlot>(&slot)) {
            container.queue_descriptor_write_image(binding, image->image, array_index,
                                                   image->access_layout);
        } else if (const auto* buffer = std::get_if<BufferHandle>(&slot)) {
            container.queue_descriptor_write_buffer(binding, *buffer, 0, VK_WHOLE_SIZE,
                                                    array_index);
        } else if (const auto* texture = std::get_if<TextureSlot>(&slot)) {
            container.queue_descriptor_write_texture(binding, texture->texture, array_index,
                                                     texture->access_layout);
        } else if (const auto* sampler = std::get_if<SamplerHandle>(&slot)) {
            container.queue_descriptor_write_sampler(binding, *sampler, array_index);
        } else if (const auto* as = std::get_if<AccelerationStructureHandle>(&slot)) {
            container.queue_descriptor_write_acceleration_structure(binding, *as, array_index);
        }
    });
}

void ShaderObject::for_each_descriptor_target(
    const std::function<void(ShaderObject&, uint32_t binding_base)>& fn) {
    for (auto it = descriptor_targets.begin(); it != descriptor_targets.end();) {
        if (auto parameter_block = it->parameter_block.lock()) {
            fn(*parameter_block, it->binding_base);
            ++it;
        } else {
            it = descriptor_targets.erase(it);
        }
    }
}

void ShaderObject::for_each_descriptor_container(
    const std::function<void(DescriptorContainer&)>& fn) {
    assert(object_layout->is_parameter_block());
    assert(descriptors);

    fn(*descriptors);
    for (auto it = registered_sets.begin(); it != registered_sets.end();) {
        if (auto set = it->lock()) {
            fn(*set);
            ++it;
        } else {
            it = registered_sets.erase(it);
        }
    }
}

// ---------------------------------------------------------------
// Writes

void ShaderObject::write(const ShaderOffset& offset, const void* data, const std::size_t size) {
    assert(object_layout->is_struct());

    const auto container = owner_container.lock();
    assert(container && "uniform write requires an owning container");
    if (container) {
        container->write_uniform(offset.uniform_byte_offset, data, size);
    }
}

void ShaderObject::write_uniform(const std::size_t byte_offset,
                                 const void* data,
                                 const std::size_t size) {
    assert(object_layout->is_container());

    const std::size_t offset =
        static_cast<std::size_t>(object_layout->get_element_uniform_offset()) + byte_offset;
    assert(offset + size <= uniform_staging.size());

    std::memcpy(uniform_staging.data() + offset, data, size);
    uniform_dirty = true;
}

void ShaderObject::write(const ShaderOffset& offset,
                         const ImageViewHandle& image,
                         const std::optional<vk::ImageLayout> access_layout) {
    assert(object_layout->is_struct());
    const auto& info = object_layout->get_binding_range_info(offset.binding_range_offset);
    assert(info.type == slang::BindingType::Texture ||
           info.type == slang::BindingType::MutableTexture);
    assert(offset.binding_array_index < info.count);

    const ResourceSlot& slot = slots[info.slot_base + offset.binding_array_index] =
        ImageSlot{image, access_layout};
    for_each_descriptor_target([&](ShaderObject& parameter_block, const uint32_t binding_base) {
        apply_slot(slot, parameter_block, binding_base + info.binding, offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset, const BufferHandle& buffer) {
    assert(object_layout->is_struct());
    const auto& info = object_layout->get_binding_range_info(offset.binding_range_offset);
    assert(info.type == slang::BindingType::RawBuffer ||
           info.type == slang::BindingType::MutableRawBuffer ||
           info.type == slang::BindingType::ConstantBuffer);
    assert(offset.binding_array_index < info.count);

    const ResourceSlot& slot = slots[info.slot_base + offset.binding_array_index] = buffer;
    for_each_descriptor_target([&](ShaderObject& parameter_block, const uint32_t binding_base) {
        apply_slot(slot, parameter_block, binding_base + info.binding, offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset,
                         const TextureHandle& texture,
                         const std::optional<vk::ImageLayout> access_layout) {
    assert(object_layout->is_struct());
    const auto& info = object_layout->get_binding_range_info(offset.binding_range_offset);
    if (info.type == slang::BindingType::Texture ||
        info.type == slang::BindingType::MutableTexture) {
        write(offset, texture->get_view(), access_layout);
        return;
    }
    if (info.type == slang::BindingType::Sampler) {
        write(offset, texture->get_sampler());
        return;
    }

    assert(info.type == slang::BindingType::CombinedTextureSampler);
    assert(offset.binding_array_index < info.count);

    const ResourceSlot& slot = slots[info.slot_base + offset.binding_array_index] =
        TextureSlot{texture, access_layout};
    for_each_descriptor_target([&](ShaderObject& parameter_block, const uint32_t binding_base) {
        apply_slot(slot, parameter_block, binding_base + info.binding, offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset, const SamplerHandle& sampler) {
    assert(object_layout->is_struct());
    const auto& info = object_layout->get_binding_range_info(offset.binding_range_offset);
    assert(info.type == slang::BindingType::Sampler);
    assert(offset.binding_array_index < info.count);

    const ResourceSlot& slot = slots[info.slot_base + offset.binding_array_index] = sampler;
    for_each_descriptor_target([&](ShaderObject& parameter_block, const uint32_t binding_base) {
        apply_slot(slot, parameter_block, binding_base + info.binding, offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset, const AccelerationStructureHandle& as) {
    assert(object_layout->is_struct());
    const auto& info = object_layout->get_binding_range_info(offset.binding_range_offset);
    assert(info.type == slang::BindingType::RayTracingAccelerationStructure);
    assert(offset.binding_array_index < info.count);

    const ResourceSlot& slot = slots[info.slot_base + offset.binding_array_index] = as;
    for_each_descriptor_target([&](ShaderObject& parameter_block, const uint32_t binding_base) {
        apply_slot(slot, parameter_block, binding_base + info.binding, offset.binding_array_index);
    });
}

// ---------------------------------------------------------------
// Binding

void ShaderObject::upload_uniform_buffers(const CommandBufferHandle& cmd) {
    assert(object_layout->is_container());

    if (uniform_buffer && uniform_dirty) {
        allocator->get_staging()->cmd_to_device(cmd, uniform_buffer, uniform_staging.data(), 0,
                                                uniform_staging.size());
        uniform_dirty = false;
    }

    if (!element) {
        return;
    }

    for (const auto& subobject : element->subobjects) {
        if (subobject && subobject->object_layout->is_constant_buffer()) {
            subobject->upload_uniform_buffers(cmd);
        }
    }
}

void ShaderObject::bind_as_parameter_block(const CommandBufferHandle& cmd,
                                           const PipelineHandle& pipeline,
                                           const uint32_t set_index,
                                           const ShaderObjectAllocatorHandle& obj_allocator) {
    assert(object_layout->is_parameter_block());
    assert(descriptors && "binding a ParameterBlock without bindings");

    upload_uniform_buffers(cmd);

    const auto container = obj_allocator->allocate(shared_from_this());

    bool found = false;
    for (auto it = registered_sets.begin(); it != registered_sets.end();) {
        if (it->expired()) {
            it = registered_sets.erase(it);
            continue;
        }
        if (it->lock() == container) {
            found = true;
        }
        ++it;
    }
    if (!found) {
        registered_sets.emplace_back(container);
        descriptors->replay_to(*container);
    }

    container->update();
    container->bind(cmd, pipeline, set_index);
}

std::string format_as(const ShaderObject& shader_object, const std::string& indent) {
    std::string out;
    if (shader_object.get_object_layout()->is_parameter_block()) {
        out += fmt::format("{}registered_sets: {}\n", indent, shader_object.registered_sets.size());
    }
    if (shader_object.get_object_layout()->is_struct()) {
        out += fmt::format("{}descriptor_targets: {}\n", indent,
                           shader_object.descriptor_targets.size());
    }
    out += fmt::format("{}shader object layout:\n", indent);
    out += format_as(*shader_object.get_object_layout(), indent + "  ");
    return out;
}

} // namespace merian
