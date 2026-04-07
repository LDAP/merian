#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object.hpp"

namespace merian {

ShaderCursor::ShaderCursor(ShaderObject* base_object)
    : base_object(base_object), type_layout(base_object->get_type_layout()) {}

ShaderCursor ShaderCursor::dereference() {
    assert(is_parameter_block() || is_constant_buffer());

    const int32_t subobject_range_index =
        base_object->get_object_layout()->find_subobject_range_index(offset.binding_range_offset);
    assert(subobject_range_index >= 0 &&
           "ConstantBuffer and ParameterBlock field must have a sub-object range");

    ShaderObjectHandle subobject = base_object->subobjects[subobject_range_index];
    if (!subobject) {
        // Auto-create the subobject from the pre-computed element layout.
        const auto& subobject_range_info =
            base_object->get_object_layout()->get_subobject_range_info(subobject_range_index);
        assert(subobject_range_info.element_layout);
        subobject = std::make_shared<ShaderObject>(subobject_range_info.element_layout,
                                                   base_object->get_allocator());
        base_object->set_subobject(subobject_range_index, subobject);
    }

    return subobject->get_cursor();
}

ShaderCursor ShaderCursor::field(const std::string& name) {
    if (!is_valid()) {
        SPDLOG_ERROR("Cannot navigate field '{}' on invalid cursor", name);
        return ShaderCursor();
    }

    // Auto-dereference PB/CB to navigate into their element type
    if (is_parameter_block() || is_constant_buffer()) {
        return dereference().field(name);
    }

    const SlangInt field_index = type_layout->findFieldIndexByName(name.c_str());
    if (field_index < 0) {
        SPDLOG_ERROR("Field '{}' not found in type {}", name, type_layout->getName());
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

    slang::VariableLayoutReflection* field_var = type_layout->getFieldByIndex(index);
    slang::TypeLayoutReflection* field_type_layout = field_var->getTypeLayout();

    ShaderCursor result;
    result.base_object = base_object;
    result.type_layout = field_type_layout;
    result.offset = offset;
    result.offset.uniform_byte_offset += field_var->getOffset();
    result.offset.binding_range_offset += type_layout->getFieldBindingRangeOffset(index);

    return result;
}

ShaderCursor ShaderCursor::element(uint32_t index) {
    if (!is_valid()) {
        SPDLOG_ERROR("Cannot navigate element {} on invalid cursor", index);
        return ShaderCursor();
    }

    // Auto-dereference PB/CB to navigate into their element type
    if (is_parameter_block() || is_constant_buffer()) {
        return dereference().element(index);
    }

    const slang::TypeReflection::Kind kind = get_kind();
    switch (kind) {
    case slang::TypeReflection::Kind::Array: {
        assert(get_element_type_layout());

        ShaderCursor result;
        result.base_object = base_object;
        result.type_layout = get_element_type_layout();
        result.offset = offset;
        result.offset.uniform_byte_offset += index * get_element_stride();
        result.offset.binding_array_index =
            (offset.binding_array_index * type_layout->getElementCount()) + index;
        return result;
    }
    case slang::TypeReflection::Kind::Vector:
    case slang::TypeReflection::Kind::Matrix: {
        assert(get_element_type_layout());

        ShaderCursor result;
        result.base_object = base_object;
        result.type_layout = get_element_type_layout();
        result.offset = offset;
        result.offset.uniform_byte_offset += index * get_element_stride();
        return result;
    }
    case slang::TypeReflection::Kind::Struct:
        return field(index);
    default:
        SPDLOG_ERROR("Type {} of kind {} cannot be accessed by element", type_layout->getName(),
                     slang_type_kind_to_string(kind));
        return ShaderCursor();
    }
}

ShaderCursor ShaderCursor::operator[](uint32_t index) {
    return element(index);
}

std::vector<std::string> ShaderCursor::get_field_names() const {
    std::vector<std::string> names;
    const uint32_t count = get_field_count();
    names.reserve(count);

    for (uint32_t i = 0; i < count; i++) {
        names.emplace_back(type_layout->getFieldByIndex(i)->getVariable()->getName());
    }

    return names;
}

ShaderCursor& ShaderCursor::write(const ImageViewHandle& image) {
    base_object->write(offset, image);
    return *this;
}

ShaderCursor& ShaderCursor::write(const BufferHandle& buffer) {
    base_object->write(offset, buffer);
    return *this;
}

ShaderCursor& ShaderCursor::write(const TextureHandle& texture) {
    base_object->write(offset, texture);
    return *this;
}

ShaderCursor& ShaderCursor::write(const SamplerHandle& sampler) {
    base_object->write(offset, sampler);
    return *this;
}

ShaderCursor& ShaderCursor::write(const AccelerationStructureHandle& as) {
    base_object->write(offset, as);
    return *this;
}

ShaderCursor& ShaderCursor::write(const void* data, std::size_t size) {
    base_object->write(offset, data, size);
    return *this;
}

ShaderCursor& ShaderCursor::write(const ShaderObjectHandle& object) {
    const slang::TypeReflection::Kind kind = get_kind();

    if (kind == slang::TypeReflection::Kind::ConstantBuffer ||
        kind == slang::TypeReflection::Kind::ParameterBlock) {
        int32_t subobject_range_index =
            base_object->get_object_layout()->find_subobject_range_index(
                offset.binding_range_offset);
        assert(subobject_range_index >= 0 &&
               "Cursor must point to a ConstantBuffer or ParameterBlock sub-object range");

        base_object->set_subobject(static_cast<uint32_t>(subobject_range_index), object);
        return *this;
    }

    throw std::runtime_error("write(ShaderObjectHandle) only supported for ConstantBuffer / "
                             "ParameterBlock cursor positions");
}

std::string format_as(const ShaderCursor& cursor) {
    if (!cursor.is_valid())
        return "(invalid cursor)";

    std::string out = fmt::format("ShaderCursor at {} of shader object:\n", cursor.get_offset());
    out += format_as(*cursor.get_base_object(), "  ");

    return out;
}

} // namespace merian
