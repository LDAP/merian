#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object.hpp"

#include <stdexcept>

namespace merian {

static std::string_view type_name_of(slang::TypeLayoutReflection* type_layout) {
    const char* const name = type_layout->getName();
    return name != nullptr ? name : "<unnamed>";
}

ShaderCursor::ShaderCursor(ShaderObject* base_object)
    : base_object(base_object), type_layout(base_object->get_type_layout()),
      kind(base_object->get_object_layout()->get_type_kind()) {
    assert(base_object->get_object_layout()->is_struct() &&
           "create cursors through ShaderObject::get_cursor()");
}

ShaderCursor ShaderCursor::dereference() {
    assert(is_parameter_block() || is_constant_buffer());

    const int32_t subobject_range_index =
        base_object->get_object_layout()->find_subobject_range_index(offset.binding_range_offset);
    assert(subobject_range_index >= 0 &&
           "ConstantBuffer and ParameterBlock field must have a sub-object range");

    ShaderObjectHandle subobject = base_object->subobjects[subobject_range_index];
    if (!subobject) {
        // Auto-create the container sub-object from the pre-computed layout.
        const auto& subobject_range_info =
            base_object->get_object_layout()->get_subobject_range_info(subobject_range_index);
        assert(subobject_range_info.container_layout);
        subobject = std::make_shared<ShaderObject>(subobject_range_info.container_layout,
                                                   base_object->get_allocator());
        base_object->set_subobject(subobject_range_index, subobject);
    }

    return subobject->get_cursor();
}

ShaderCursor ShaderCursor::field(const std::string_view name) {
    if (!is_valid()) {
        throw std::invalid_argument{fmt::format("field '{}' on an invalid cursor", name)};
    }

    if (is_parameter_block() || is_constant_buffer()) {
        return dereference().field(name);
    }

    const ShaderStructFields& struct_fields = get_struct_fields();
    const int32_t field_index = struct_fields.find(name);
    if (field_index < 0) {
        throw std::invalid_argument{
            fmt::format("field '{}' not found in type {}", name, type_name_of(type_layout))};
    }
    return field(struct_fields, static_cast<uint32_t>(field_index));
}

ShaderCursor ShaderCursor::find(const std::string_view name) {
    if (!is_valid()) {
        return ShaderCursor();
    }
    if (is_parameter_block() || is_constant_buffer()) {
        return dereference().find(name);
    }
    const ShaderStructFields& struct_fields = get_struct_fields();
    const int32_t field_index = struct_fields.find(name);
    return field_index < 0 ? ShaderCursor()
                           : field(struct_fields, static_cast<uint32_t>(field_index));
}

bool ShaderCursor::has_field(const std::string_view name) const {
    return get_struct_fields().find(name) >= 0;
}

ShaderCursor ShaderCursor::field(const ShaderStructFields& struct_fields,
                                 const uint32_t index) const {
    assert(index < struct_fields.fields.size());
    const ShaderStructFields::Field& struct_field = struct_fields.fields[index];

    ShaderCursor result;
    result.base_object = base_object;
    result.type_layout = struct_field.type_layout;
    result.kind = struct_field.kind;
    result.offset = offset;
    result.offset.uniform_byte_offset += struct_field.uniform_offset;
    result.offset.binding_range_offset += struct_field.binding_range_offset;

    return result;
}

const ShaderStructFields& ShaderCursor::get_struct_fields() const {
    return base_object->get_object_layout()->get_program()->get_struct_fields(type_layout);
}

ShaderCursor ShaderCursor::element(uint32_t index) {
    if (!is_valid()) {
        throw std::invalid_argument{fmt::format("element {} on an invalid cursor", index)};
    }

    if (is_parameter_block() || is_constant_buffer()) {
        return dereference().element(index);
    }

    switch (kind) {
    case slang::TypeReflection::Kind::Array: {
        assert(get_element_type_layout());
        assert(index < get_element_count());

        ShaderCursor result;
        result.base_object = base_object;
        result.type_layout = get_element_type_layout();
        result.kind = result.type_layout->getType()->getKind();
        result.offset = offset;
        result.offset.uniform_byte_offset += index * get_element_stride();
        result.offset.binding_array_index =
            (offset.binding_array_index * type_layout->getElementCount()) + index;
        return result;
    }
    case slang::TypeReflection::Kind::Vector:
    case slang::TypeReflection::Kind::Matrix: {
        assert(get_element_type_layout());
        assert(index < get_element_count());

        ShaderCursor result;
        result.base_object = base_object;
        result.type_layout = get_element_type_layout();
        result.kind = result.type_layout->getType()->getKind();
        result.offset = offset;
        result.offset.uniform_byte_offset += index * get_element_stride();
        return result;
    }
    case slang::TypeReflection::Kind::Struct:
        return field(get_struct_fields(), index);
    default:
        throw std::invalid_argument{fmt::format("type {} of kind {} cannot be accessed by element",
                                                type_name_of(type_layout),
                                                slang_type_kind_to_string(kind))};
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

ShaderCursor& ShaderCursor::write(const ImageViewHandle& image,
                                  const std::optional<vk::ImageLayout> access_layout) {
    base_object->write(offset, image, access_layout);
    return *this;
}

ShaderCursor& ShaderCursor::write(const BufferHandle& buffer) {
    base_object->write(offset, buffer);
    return *this;
}

ShaderCursor& ShaderCursor::write(const TextureHandle& texture,
                                  const std::optional<vk::ImageLayout> access_layout) {
    base_object->write(offset, texture, access_layout);
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
    if (is_constant_buffer() || is_parameter_block()) {
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
