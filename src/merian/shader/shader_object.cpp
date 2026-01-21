#include "merian/shader/shader_object.hpp"

#include <cassert>
#include <spdlog/spdlog.h>

namespace merian {

ShaderObject::ShaderObject(const ContextHandle& ctx, slang::TypeLayoutReflection* layout)
    : type_layout(layout), context(ctx) {
    assert(type_layout);
}

DescriptorContainerHandle
ShaderObject::initialize_as_parameter_block(ShaderObjectAllocator& allocator) {
    // // Get or create descriptor set layout from Slang reflection
    // auto layout = create_descriptor_set_layout_from_slang(context, type_layout);

    // // Get or allocate descriptor set
    // descriptor_set = allocator.get_or_create_descriptor_set(shared_from_this(), layout);

    // // Allocate ordinary data buffer if needed
    // size_t ordinary_size = type_layout->getSize();
    // if (ordinary_size > 0) {
    //     vk::BufferCreateInfo buffer_info{{},
    //                                      ordinary_size,
    //                                      vk::BufferUsageFlagBits::eUniformBuffer |
    //                                          vk::BufferUsageFlagBits::eTransferDst,
    //                                      vk::SharingMode::eExclusive};

    //     // TODO: Use project-specific buffer creation method
    //     // ordinary_data = create_buffer(...);
    //     ordinary_data_staging.resize(ordinary_size);

    //     SPDLOG_DEBUG("Allocated ordinary data buffer of size {} for ShaderObject {}",
    //     ordinary_size,
    //                  fmt::ptr(this));
    // }

    // // Initialize root cursor with this single location
    // root_cursor = ShaderCursor(shared_from_this());

    // // Populate with initial data
    // populate(*root_cursor);

    // // Update descriptor set and buffer
    // if (ordinary_data) {
    //     // TODO: Upload staging buffer to GPU
    //     // map, memcpy, unmap
    // }
    // descriptor_set->update();

    // return descriptor_set;
}

void ShaderObject::bind_to(ShaderCursor& cursor, ShaderObjectAllocator& allocator) {
    // auto kind = cursor.get_kind();

    // if (kind == slang::TypeReflection::Kind::ParameterBlock) {
    //     // Initialize as parameter block if not already done
    //     if (!descriptor_set) {
    //         initialize_as_parameter_block(allocator);
    //     }
    //     // Nested parameter blocks are bound separately at draw time

    // } else if (kind == slang::TypeReflection::Kind::ConstantBuffer) {
    //     // Allocate our own buffer if needed
    //     if (!ordinary_data) {
    //         size_t buffer_size = type_layout->getSize();

    //         vk::BufferCreateInfo buffer_info{{},
    //                                          buffer_size,
    //                                          vk::BufferUsageFlagBits::eUniformBuffer |
    //                                              vk::BufferUsageFlagBits::eTransferDst,
    //                                          vk::SharingMode::eExclusive};

    //         // TODO: Use project-specific buffer creation method
    //         // ordinary_data = create_buffer(...);
    //         ordinary_data_staging.resize(buffer_size);
    //     }

    //     // Bind our buffer to all parent descriptor sets
    //     for (size_t i = 0; i < cursor.locations.size(); i++) {
    //         auto& loc = cursor.locations[i];
    //         auto binding_info =
    //             get_binding_info_from_offset(loc.offset, loc.base_object->get_type_layout());
    //         auto parent_desc_set = loc.base_object->get_descriptor_set();

    //         if (parent_desc_set) {
    //             parent_desc_set->queue_descriptor_write_buffer(binding_info.binding,
    //             ordinary_data,
    //                                                            0, VK_WHOLE_SIZE,
    //                                                            loc.offset.binding_array_index);
    //         }
    //     }

    //     // Initialize our root cursor if empty
    //     if (!root_cursor) {
    //         root_cursor = ShaderCursor(shared_from_this(), type_layout);
    //     }

    //     // Populate our buffer
    //     populate(*root_cursor);

    //     // Upload and update parent descriptor sets
    //     if (ordinary_data) {
    //         // TODO: Upload buffer
    //     }

    //     for (auto& loc : cursor.locations) {
    //         if (auto parent_desc_set = loc.base_object->get_descriptor_set()) {
    //             parent_desc_set->update();
    //         }
    //     }

    // } else {
    //     // Nested as value - add all cursor locations to our root cursor
    //     if (!root_cursor) {
    //         // First binding - adopt the cursor locations
    //         root_cursor = cursor;
    //     } else {
    //         // Additional binding - merge locations
    //         root_cursor->add_locations(cursor);
    //     }

    //     // Populate through the cursor
    //     populate(cursor);
    // }
}

void ShaderObject::for_each_descriptor_set(
    const std::function<void(const DescriptorContainerHandle&)>& f) {
    auto it = parameter_block.descriptor_sets.begin();
    while (it != parameter_block.descriptor_sets.end()) {
        if (it->expired()) {
            it = parameter_block.descriptor_sets.erase(it);
            continue;
        }

        f(it->lock());
    }
}

void ShaderObject::write(const ShaderOffset& offset, const ImageViewHandle& image) {
    const auto binding_info = get_binding_info_from_offset(offset, type_layout);
    for_each_descriptor_set([&](const DescriptorContainerHandle& set) {
        set->queue_descriptor_write_image(binding_info.binding, image, offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset, const BufferHandle& buffer) {
    const auto binding_info = get_binding_info_from_offset(offset, type_layout);
    for_each_descriptor_set([&](const DescriptorContainerHandle& set) {
        set->queue_descriptor_write_buffer(binding_info.binding, buffer,
                                           offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset, const TextureHandle& texture) {
    const auto binding_info = get_binding_info_from_offset(offset, type_layout);
    for_each_descriptor_set([&](const DescriptorContainerHandle& set) {
        set->queue_descriptor_write_texture(binding_info.binding, texture,
                                            offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset, const SamplerHandle& sampler) {
    const auto binding_info = get_binding_info_from_offset(offset, type_layout);
    for_each_descriptor_set([&](const DescriptorContainerHandle& set) {
        set->queue_descriptor_write_sampler(binding_info.binding, sampler,
                                            offset.binding_array_index);
    });
}

void ShaderObject::write(const ShaderOffset& offset, const void* data, std::size_t size) {
    // Write to our staging buffer if we have one
    if (!parameter_block.ordinary_data_staging.empty()) {
        assert(offset.uniform_byte_offset + size <= parameter_block.ordinary_data_staging.size());
        std::memcpy(parameter_block.ordinary_data_staging.data() + offset.uniform_byte_offset, data,
                    size);
    }

    // TODO: mark dirty and upload to GPU later (when binding)
}

} // namespace merian
