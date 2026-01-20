#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object.hpp"

namespace merian {

ShaderCursor::ShaderCursor(const ShaderObjectHandle& base_object)
    : type_layout(base_object->get_type_layout()) {
    locations.emplace_back(base_object, ShaderOffset());
}

ShaderCursor ShaderCursor::field(const std::string& name) {
    if (!is_valid()) {
        SPDLOG_ERROR("Cannot navigate field '{}' on invalid cursor", name);
        return ShaderCursor();
    }

    SlangInt field_index = type_layout->findFieldIndexByName(name.c_str());
    if (field_index < 0) {
        SPDLOG_ERROR("Field '{}' not found in type", name);
        return ShaderCursor();
    }

    return field(static_cast<uint32_t>(field_index));
}

ShaderCursor ShaderCursor::field(uint32_t index) {
    if (!is_valid()) {
        SPDLOG_ERROR("Cannot navigate field {} on invalid cursor", index);
        return ShaderCursor();
    }

    assert(index < type_layout->getFieldCount());

    slang::VariableLayoutReflection* field = type_layout->getFieldByIndex(index);

    ShaderCursor result;
    result.type_layout = field->getTypeLayout();
    result.locations.reserve(locations.size());

    for (auto& loc : locations) {
        ShaderOffset new_offset = loc.offset;
        new_offset.uniform_byte_offset += field->getOffset();
        new_offset.binding_range_offset += type_layout->getFieldBindingRangeOffset(index);

        result.locations.push_back({loc.base_object, new_offset});
    }

    return result;
}

ShaderCursor ShaderCursor::element(uint32_t index) {
    if (!is_valid()) {
        SPDLOG_ERROR("Cannot navigate element {} on invalid cursor", index);
        return ShaderCursor();
    }

    slang::TypeLayoutReflection* element_type_layout = type_layout->getElementTypeLayout();
    if (element_type_layout == nullptr) {
        SPDLOG_ERROR("Type is not an array, cannot access element {}", index);
        return ShaderCursor();
    }

    ShaderCursor result;
    result.type_layout = element_type_layout;
    result.locations.reserve(locations.size());

    for (auto& loc : locations) {
        ShaderOffset new_offset = loc.offset;
        new_offset.uniform_byte_offset += index * element_type_layout->getStride();
        new_offset.binding_array_index =
            (loc.offset.binding_array_index * type_layout->getElementCount()) + index;

        result.locations.push_back({loc.base_object, new_offset});
    }

    return result;
}

ShaderCursor ShaderCursor::operator[](uint32_t index) {
    if (type_layout->getType()->getKind() == slang::TypeReflection::Kind::Array) {
        return element(index);
    }

    return field(index);
}

ShaderCursor& ShaderCursor::write(const ImageViewHandle& image) {
    for (auto& loc : locations) {
        loc.base_object->write(loc.offset, image);
    }
    return *this;
}

ShaderCursor& ShaderCursor::write(const BufferHandle& buffer) {
    for (auto& loc : locations) {
        loc.base_object->write(loc.offset, buffer);
    }
    return *this;
}

ShaderCursor& ShaderCursor::write(const TextureHandle& texture) {
    for (auto& loc : locations) {
        loc.base_object->write(loc.offset, texture);
    }
    return *this;
}

ShaderCursor& ShaderCursor::write(const SamplerHandle& sampler) {
    for (auto& loc : locations) {
        loc.base_object->write(loc.offset, sampler);
    }
    return *this;
}

ShaderCursor& ShaderCursor::write(const void* data, std::size_t size) {
    for (auto& loc : locations) {
        loc.base_object->write(loc.offset, data, size);
    }
    return *this;
}

void ShaderCursor::bind_object(const ShaderObjectHandle& object, ShaderObjectAllocator& allocator) {
    // if (!is_valid()) {
    //     SPDLOG_ERROR("Cannot bind object to invalid cursor");
    //     return;
    // }

    // auto kind = type_layout->getType()->getKind();

    // if (kind == slang::TypeReflection::Kind::ParameterBlock) {
    //     // Nested parameter block - gets its own descriptor set
    //     if (!object->get_descriptor_set()) {
    //         object->initialize_as_parameter_block(allocator);
    //     }
    //     // The nested parameter block's descriptor set is bound separately at draw time

    // } else {
    //     // Constant buffer or value - bind to all our locations
    //     object->bind_to(*this, allocator);
    // }
}

void ShaderCursor::add_locations(const ShaderCursor& other) {
    // assert(type_layout == other.type_layout || !type_layout);
    // if (!type_layout) {
    //     type_layout = other.type_layout;
    // }
    // locations.insert(locations.end(), other.locations.begin(), other.locations.end());
}

} // namespace merian
