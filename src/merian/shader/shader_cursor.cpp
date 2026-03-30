#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object.hpp"

namespace merian {

ShaderCursor::ShaderCursor(const ShaderObjectHandle& base_object)
    : base_object(base_object), type_layout(base_object->get_type_layout()) {}

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

    slang::VariableLayoutReflection* field_var = type_layout->getFieldByIndex(index);
    auto* field_type_layout = field_var->getTypeLayout();
    auto field_kind = field_type_layout->getKind();

    // For ConstantBuffer<T> and ParameterBlock<T> fields: auto-create the sub-object
    // if it doesn't exist, and return a cursor into the sub-object.
    if (field_kind == slang::TypeReflection::Kind::ConstantBuffer ||
        field_kind == slang::TypeReflection::Kind::ParameterBlock) {

        // Compute the binding range for this field, accounting for offset accumulation
        // through value-embedded structs.
        uint32_t br = offset.binding_range_offset + type_layout->getFieldBindingRangeOffset(index);

        // Look up the sub-object range in the root object's layout
        int32_t sor = base_object->get_object_layout()->find_sub_object_range(br);
        assert(sor >= 0 && "CB/PB field must have a sub-object range");

        auto& sub = base_object->sub_objects[sor];
        if (!sub) {
            // Auto-create using the pre-computed element layout
            const auto& range_info = base_object->get_object_layout()->get_sub_object_range(sor);
            assert(range_info.element_layout);
            sub = std::make_shared<ShaderObject>(range_info.element_layout,
                                                 base_object->get_allocator());
            // set_sub_object handles CB descriptor writes to the owning PB
            base_object->set_sub_object(sor, sub);
        }

        // Return cursor into the sub-object (dereferenced to element type T)
        return sub->get_cursor();
    }

    // For value fields: adjust offset and keep the same base_object
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

    slang::TypeLayoutReflection* element_type_layout = type_layout->getElementTypeLayout();
    if (element_type_layout == nullptr) {
        SPDLOG_ERROR("Type is not an array, cannot access element {}", index);
        return ShaderCursor();
    }

    ShaderCursor result;
    result.base_object = base_object;
    result.type_layout = element_type_layout;
    result.offset = offset;
    result.offset.uniform_byte_offset += index * element_type_layout->getStride();
    result.offset.binding_array_index =
        (offset.binding_array_index * type_layout->getElementCount()) + index;

    return result;
}

ShaderCursor ShaderCursor::operator[](uint32_t index) {
    if (type_layout->getType()->getKind() == slang::TypeReflection::Kind::Array) {
        return element(index);
    }

    return field(index);
}

std::vector<std::string> ShaderCursor::get_field_names() const {
    std::vector<std::string> names;
    const uint32_t count = type_layout->getFieldCount();
    names.reserve(count);
    for (uint32_t i = 0; i < count; i++) {
        names.emplace_back(type_layout->getFieldByIndex(i)->getVariable()->getName());
    }
    return names;
}

std::optional<ShaderCursor> ShaderCursor::try_field(const std::string& name) {
    if (!is_valid()) {
        return std::nullopt;
    }
    SlangInt idx = type_layout->findFieldIndexByName(name.c_str());
    if (idx < 0) {
        return std::nullopt;
    }
    return field(static_cast<uint32_t>(idx));
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

ShaderCursor& ShaderCursor::write(const ShaderObjectHandle& object) {
    // Determine what kind of field this cursor points at
    auto kind = type_layout->getKind();

    if (kind == slang::TypeReflection::Kind::ConstantBuffer ||
        kind == slang::TypeReflection::Kind::ParameterBlock) {
        // Find the sub-object range for this CB/PB in the root object's layout
        int32_t sor =
            base_object->get_object_layout()->find_sub_object_range(offset.binding_range_offset);
        assert(sor >= 0 && "CB/PB cursor must map to a sub-object range");

        // set_sub_object handles CB descriptor writes to the owning PB
        base_object->set_sub_object(static_cast<uint32_t>(sor), object);
    } else {
        SPDLOG_ERROR("write(ShaderObjectHandle) only supported for ConstantBuffer/ParameterBlock "
                     "cursor positions");
    }

    return *this;
}

ShaderCursor& ShaderCursor::write(const void* data, std::size_t size) {
    base_object->write(offset, data, size);
    return *this;
}

std::string ShaderCursor::format_debug() const {
    if (!is_valid())
        return "(invalid cursor)\n";

    std::string out;
    const char* name = type_layout->getName();
    out += fmt::format("ShaderCursor at {}\n", format_shader_offset(offset));
    out += fmt::format("  type: '{}', kind={}\n", name ? name : "(anonymous)",
                       slang_type_kind_to_string(type_layout->getKind()));
    out += fmt::format("  uniform_size: {} bytes\n",
                       type_layout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM));

    uint32_t field_count = type_layout->getFieldCount();
    if (field_count > 0) {
        out += fmt::format("  fields ({}):\n", field_count);
        for (uint32_t f = 0; f < field_count; f++) {
            auto* fv = type_layout->getFieldByIndex(f);
            auto* ftl = fv->getTypeLayout();
            const char* fname = fv->getVariable()->getName();
            out += fmt::format("    [{}] '{}': kind={}, uniform_offset={}, "
                               "binding_range_offset={}\n",
                               f, fname ? fname : "?", slang_type_kind_to_string(ftl->getKind()),
                               fv->getOffset(SLANG_PARAMETER_CATEGORY_UNIFORM),
                               type_layout->getFieldBindingRangeOffset(f));
        }
    }

    if (type_layout->getKind() == slang::TypeReflection::Kind::Array) {
        out += fmt::format("  array: element_count={}, stride={}\n", type_layout->getElementCount(),
                           type_layout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM));
    }

    return out;
}

} // namespace merian
