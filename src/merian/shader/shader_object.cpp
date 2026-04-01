#include "merian/shader/shader_object.hpp"

#include <cassert>
#include <spdlog/spdlog.h>

namespace merian {

ShaderObject::ShaderObject(const ShaderObjectLayoutHandle& object_layout,
                           const ShaderObjectAllocatorHandle& allocator)
    : object_layout(object_layout), allocator(allocator) {
    assert(object_layout);
    assert(allocator);

    // Create descriptor storage for caching writes (incremental update model)
    descriptors = DescriptorStorage::create(object_layout->get_descriptor_set_layout());

    // Pre-size sub-object array: one slot per CB/PB field
    subobjects.resize(object_layout->get_subobject_range_count());

    // Eagerly create ordinary data buffer if needed
    const vk::DeviceSize uniform_size = object_layout->get_uniform_size();
    if (uniform_size > 0) {
        ordinary_data_staging.resize(uniform_size, 0);
        ordinary_data_buffer = allocator->allocate_uniform_buffer(uniform_size);

        // Write buffer to descriptor storage so it gets replayed to new sets
        descriptors->queue_descriptor_write_buffer(ShaderObjectLayout::ORDINARY_DATA_BUFFER_BINDING,
                                                   ordinary_data_buffer, 0, uniform_size);
    }
}

// ---------------------------------------------------------------
// Cursor

ShaderCursor ShaderObject::get_cursor() {
    return ShaderCursor(shared_from_this());
}

// ---------------------------------------------------------------
// Binding

// Compute the sub-range offset, container offset (UBO binding), and element offset
// for a ConstantBuffer sub-object range within a parent type layout.
// Returns {ubo_binding_delta, element_binding_delta} relative to parent's base binding.
static std::pair<uint32_t, uint32_t> compute_cb_binding_deltas(
    slang::TypeLayoutReflection* parent_tl, uint32_t sor, uint32_t binding_range_index) {
    uint32_t sub_range_offset = 0;
    if (auto* range_var = parent_tl->getSubObjectRangeOffset(sor)) {
        sub_range_offset = static_cast<uint32_t>(
            range_var->getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
    }

    auto* leaf_tl = parent_tl->getBindingRangeLeafTypeLayout(binding_range_index);
    assert(leaf_tl && "CB sub-object range must have leaf type layout");

    uint32_t container_offset = 0;
    if (auto* cv = leaf_tl->getContainerVarLayout()) {
        container_offset =
            static_cast<uint32_t>(cv->getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
    }

    uint32_t element_offset = 0;
    if (auto* ev = leaf_tl->getElementVarLayout()) {
        element_offset =
            static_cast<uint32_t>(ev->getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
    }

    return {sub_range_offset + container_offset, sub_range_offset + element_offset};
}

void ShaderObject::set_subobject(uint32_t subobject_range_index, const ShaderObjectHandle& object) {
    assert(subobject_range_index < subobjects.size());
    subobjects[subobject_range_index] = object;

    const auto& range = object_layout->get_subobject_range_info(subobject_range_index);
    if (range.binding_type != slang::BindingType::ConstantBuffer || !object ||
        !object->ordinary_data_buffer) {
        return;
    }

    // For CB sub-objects: write the UBO descriptor to the owning PB's storage/sets
    // and set the CB's PB context for nested CB propagation.
    auto* tl = object_layout->get_type_layout();
    auto [ubo_delta, element_delta] =
        compute_cb_binding_deltas(tl, subobject_range_index, range.binding_range_index);

    if (!cb_owners.empty()) {
        // This object is a CB inside one or more PBs — propagate to all owning PBs.
        for (auto it = cb_owners.begin(); it != cb_owners.end();) {
            auto pb = it->pb.lock();
            if (!pb) {
                it = cb_owners.erase(it);
                continue;
            }

            uint32_t ubo_binding = it->element_binding + ubo_delta;
            uint32_t element_binding = it->element_binding + element_delta;

            pb->descriptors->queue_descriptor_write_buffer(
                ubo_binding, object->ordinary_data_buffer, 0, VK_WHOLE_SIZE);
            pb->for_each_registered_set([&](DescriptorContainer& set) {
                set.queue_descriptor_write_buffer(ubo_binding, object->ordinary_data_buffer, 0,
                                                  VK_WHOLE_SIZE);
            });

            object->cb_owners.push_back(OwnerConstantBufferBindings{it->pb, element_binding});
            ++it;
        }
    } else {
        // This object IS the PB. Use binding_info_cache for the Vulkan binding (same path
        // the old cursor code used via write()), and compute element_binding from reflection
        // for nested CB propagation.
        const auto& info = object_layout->get_binding_info(range.binding_range_index);
        descriptors->queue_descriptor_write_buffer(info.binding, object->ordinary_data_buffer, 0,
                                                   VK_WHOLE_SIZE);
        for_each_registered_set([&](DescriptorContainer& set) {
            set.queue_descriptor_write_buffer(info.binding, object->ordinary_data_buffer, 0,
                                              VK_WHOLE_SIZE);
        });

        uint32_t pb_element_offset = object_layout->has_ordinary_data_buffer() ? 1 : 0;
        object->cb_owners.push_back(
            OwnerConstantBufferBindings{shared_from_this(), pb_element_offset + element_delta});
    }
}

void ShaderObject::upload_constant_buffer_tree(ShaderObject* cb_obj,
                                               const CommandBufferHandle& cmd) {
    // Upload this CB's staging data if dirty
    if (cb_obj->ordinary_data_buffer && cb_obj->ordinary_data_dirty) {
        allocator->get_staging()->cmd_to_device(cmd, cb_obj->ordinary_data_buffer,
                                                cb_obj->ordinary_data_staging.data(), 0,
                                                cb_obj->ordinary_data_staging.size());
        cb_obj->ordinary_data_dirty = false;
    }

    // Recurse into nested CB sub-objects for staging uploads
    for (uint32_t sor = 0; sor < cb_obj->object_layout->get_subobject_range_count(); sor++) {
        const auto& range = cb_obj->object_layout->get_subobject_range_info(sor);
        if (range.binding_type != slang::BindingType::ConstantBuffer)
            continue;

        auto& nested = cb_obj->subobjects[sor];
        if (!nested || !nested->ordinary_data_buffer)
            continue;

        upload_constant_buffer_tree(nested.get(), cmd);
    }
}

void ShaderObject::for_each_registered_set(const std::function<void(DescriptorContainer&)>& fn) {
    for (auto it = registered_sets.begin(); it != registered_sets.end();) {
        if (auto set = it->lock()) {
            fn(*set);
            ++it;
        } else {
            it = registered_sets.erase(it);
        }
    }
}

void ShaderObject::bind_as_parameter_block(const CommandBufferHandle& cmd,
                                           const PipelineHandle& pipeline,
                                           const uint32_t set_index) {
    // Ask allocator for a descriptor container (handles frame cycling)
    auto container = allocator->allocate(this);

    // If new container, register and replay cached descriptor state
    bool found = false;
    for (auto it = registered_sets.begin(); it != registered_sets.end();) {
        if (it->expired()) {
            it = registered_sets.erase(it);
            continue;
        }
        if (it->lock() == container)
            found = true;
        ++it;
    }
    if (!found) {
        registered_sets.emplace_back(container);
        descriptors->replay_to(*container);
    }

    // Upload this PB's own ordinary data
    if (ordinary_data_buffer && ordinary_data_dirty) {
        allocator->get_staging()->cmd_to_device(cmd, ordinary_data_buffer,
                                                ordinary_data_staging.data(), 0,
                                                ordinary_data_staging.size());
        ordinary_data_dirty = false;
    }

    // Upload staging data for ConstantBuffer sub-objects (recursive, dirty-guarded).
    // Descriptor writes are handled by set_subobject at update time.
    for (uint32_t sor = 0; sor < object_layout->get_subobject_range_count(); sor++) {
        const auto& range = object_layout->get_subobject_range_info(sor);
        if (range.binding_type != slang::BindingType::ConstantBuffer)
            continue;

        auto& sub = subobjects[sor];
        if (!sub)
            continue;

        upload_constant_buffer_tree(sub.get(), cmd);
    }

    // Flush queued descriptor writes and bind
    container->update();
    container->bind(cmd, pipeline, set_index);

    // ParameterBlock sub-objects are bound by SlangProgramEntryPoint::bind_nested_pbs
}

// ---------------------------------------------------------------
// Sub-object creation

ShaderObjectHandle ShaderObject::create_subobject(const std::string& field_name) {
    auto* tl = object_layout->get_type_layout();
    SlangInt field_index = tl->findFieldIndexByName(field_name.c_str());
    assert(field_index >= 0 && "Field not found");

    // Find the sub-object range for this field
    uint32_t br = tl->getFieldBindingRangeOffset(static_cast<uint32_t>(field_index));
    int32_t sor = object_layout->find_subobject_range_index(br);
    assert(sor >= 0 && "Field must be a ConstantBuffer or ParameterBlock");

    const auto& range_info = object_layout->get_subobject_range_info(sor);
    assert(range_info.element_layout);

    return std::make_shared<ShaderObject>(range_info.element_layout, allocator);
}

ShaderObject::~ShaderObject() {
    allocator->free(this);
}

void ShaderObject::set_subobject(const std::string& field_name, const ShaderObjectHandle& object) {
    auto* tl = object_layout->get_type_layout();
    SlangInt field_index = tl->findFieldIndexByName(field_name.c_str());
    assert(field_index >= 0 && "Field not found");

    int32_t sor = object_layout->find_subobject_range_index(
        tl->getFieldBindingRangeOffset(static_cast<uint32_t>(field_index)));
    assert(sor >= 0 && "Field must be a ConstantBuffer or ParameterBlock");

    // Delegate to the sor-based version which handles CB descriptor writes
    set_subobject(static_cast<uint32_t>(sor), object);
}

// ---------------------------------------------------------------
// Write operations

void ShaderObject::write(const ShaderOffset& offset, const ImageViewHandle& image) {
    const auto& info = object_layout->get_binding_info(offset.binding_range_offset);
    assert(info.type == slang::BindingType::Texture ||
           info.type == slang::BindingType::MutableTexture);
    assert(offset.binding_array_index < info.count);

    descriptors->queue_descriptor_write_image(info.binding, image, offset.binding_array_index);
    for_each_registered_set([&](DescriptorContainer& set) {
        set.queue_descriptor_write_image(info.binding, image, offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset, const BufferHandle& buffer) {
    const auto& info = object_layout->get_binding_info(offset.binding_range_offset);
    assert(info.type == slang::BindingType::RawBuffer ||
           info.type == slang::BindingType::MutableRawBuffer ||
           info.type == slang::BindingType::ConstantBuffer);
    assert(offset.binding_array_index < info.count);

    descriptors->queue_descriptor_write_buffer(info.binding, buffer, 0, VK_WHOLE_SIZE,
                                               offset.binding_array_index);
    for_each_registered_set([&](DescriptorContainer& set) {
        set.queue_descriptor_write_buffer(info.binding, buffer, 0, VK_WHOLE_SIZE,
                                          offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset, const TextureHandle& texture) {
    const auto& info = object_layout->get_binding_info(offset.binding_range_offset);
    assert(info.type == slang::BindingType::CombinedTextureSampler);
    assert(offset.binding_array_index < info.count);

    descriptors->queue_descriptor_write_texture(info.binding, texture, offset.binding_array_index);
    for_each_registered_set([&](DescriptorContainer& set) {
        set.queue_descriptor_write_texture(info.binding, texture, offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset, const SamplerHandle& sampler) {
    const auto& info = object_layout->get_binding_info(offset.binding_range_offset);
    assert(info.type == slang::BindingType::Sampler);
    assert(offset.binding_array_index < info.count);

    descriptors->queue_descriptor_write_sampler(info.binding, sampler, offset.binding_array_index);
    for_each_registered_set([&](DescriptorContainer& set) {
        set.queue_descriptor_write_sampler(info.binding, sampler, offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset, const void* data, const std::size_t size) {
    if (!ordinary_data_staging.empty()) {
        assert(offset.uniform_byte_offset + size <= ordinary_data_staging.size());
        std::memcpy(ordinary_data_staging.data() + offset.uniform_byte_offset, data, size);
        ordinary_data_dirty = true;
    }
}

std::string format_as(const ShaderObject& shader_object, const std::string& indent) {
    std::string out;
    out += fmt::format("{}registered_sets: {}\n", indent, shader_object.registered_sets.size());
    out +=
        fmt::format("{}owned as constant buffer by: {}\n", indent, shader_object.cb_owners.size());
    out += fmt::format("{}shader object layout: \n", indent);
    out += format_as(*shader_object.get_object_layout(), indent + "  ");
    return out;
}

} // namespace merian
