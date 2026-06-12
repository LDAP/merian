#include "merian/shader/shader_object_layout.hpp"
#include "merian/shader/slang_utils.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"

namespace merian {

// Resolves SLANG_UNKNOWN_SIZE descriptor counts (link-time constant array sizes).
static uint32_t resolve_descriptor_count(slang::TypeLayoutReflection* type_layout,
                                         const uint32_t binding_range_index,
                                         const SlangInt raw_count,
                                         slang::ProgramLayout* program_layout) {
    if (raw_count != static_cast<SlangInt>(SLANG_UNKNOWN_SIZE)) {
        return static_cast<uint32_t>(raw_count);
    }

    auto* leaf_var = type_layout->getBindingRangeLeafVariable(binding_range_index);
    assert(leaf_var);
    auto* leaf_type = leaf_var->getType();
    assert(leaf_type && leaf_type->isArray());
    const size_t resolved =
        leaf_type->getElementCount(reinterpret_cast<SlangReflection*>(program_layout));
    if (resolved == SLANG_UNKNOWN_SIZE || resolved == SLANG_UNBOUNDED_SIZE) {
        throw std::runtime_error(
            "descriptor count depends on an unresolved link-time constant (extern static "
            "const). Ensure the constant is defined in the composition.");
    }
    return static_cast<uint32_t>(resolved);
}

ShaderObjectLayout::ShaderObjectLayout(const ContextHandle& context,
                                       slang::TypeLayoutReflection* type_layout,
                                       const SlangProgramHandle& program,
                                       const bool as_scope_container)
    : type_layout(type_layout), program(program), context(context) {
    assert(type_layout);
    assert(program);

    switch (type_layout->getKind()) {
    case slang::TypeReflection::Kind::ParameterBlock:
        kind = Kind::PARAMETER_BLOCK;
        init_container(context);
        break;
    case slang::TypeReflection::Kind::ConstantBuffer:
        kind = as_scope_container ? Kind::PARAMETER_BLOCK : Kind::CONSTANT_BUFFER;
        init_container(context);
        break;
    default:
        if (as_scope_container) {
            // Global scope without uniform data: the struct's bindings are the set itself.
            kind = Kind::PARAMETER_BLOCK;
            init_container(context);
        } else {
            kind = Kind::STRUCT;
            init_struct(context);
        }
        break;
    }
}

void ShaderObjectLayout::init_container(const ContextHandle& context) {
    slang::VariableLayoutReflection* element_var_layout = type_layout->getElementVarLayout();

    if (element_var_layout != nullptr) {
        element_layout =
            program->get_or_create_object_layout(context, element_var_layout->getTypeLayout());
        element_uniform_offset = element_var_layout->getOffset(SLANG_PARAMETER_CATEGORY_UNIFORM);
        element_binding_offset = static_cast<uint32_t>(
            element_var_layout->getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));

        if (auto* container_var_layout = type_layout->getContainerVarLayout()) {
            uniform_buffer_binding = static_cast<uint32_t>(
                container_var_layout->getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
        }
    } else {
        // Scope container around a plain struct: the element is the type itself, no offsets.
        element_layout = program->get_or_create_object_layout(context, type_layout);
    }

    uniform_size = element_layout->get_uniform_size();

    bool any_descriptor_range = uniform_size > 0;
    for (const auto& range : element_layout->binding_ranges) {
        if (any_descriptor_range) {
            break;
        }
        any_descriptor_range = range.type != slang::BindingType::ParameterBlock &&
                               range.type != slang::BindingType::PushConstant;
    }
    parameter_block_has_bindings = any_descriptor_range;
}

void ShaderObjectLayout::init_struct(const ContextHandle& context) {
    uniform_size = type_layout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);

    auto* program_layout = program->get_program_reflection();

    const uint32_t binding_range_count = type_layout->getBindingRangeCount();
    binding_ranges.resize(binding_range_count);
    for (uint32_t binding_range_index = 0; binding_range_index < binding_range_count;
         binding_range_index++) {
        const slang::BindingType binding_type =
            type_layout->getBindingRangeType(binding_range_index);
        const uint32_t count = resolve_descriptor_count(
            type_layout, binding_range_index,
            type_layout->getBindingRangeBindingCount(binding_range_index), program_layout);

        // ParameterBlock binding ranges have no descriptors in this set.
        uint32_t binding = 0;
        const SlangInt first_descriptor_range =
            type_layout->getBindingRangeFirstDescriptorRangeIndex(binding_range_index);
        if (first_descriptor_range >= 0) {
            binding = static_cast<uint32_t>(
                type_layout->getDescriptorSetDescriptorRangeIndexOffset(0, first_descriptor_range));
        }

        binding_ranges[binding_range_index] =
            BindingRangeInfo{binding, binding_type, count, slot_count};
        slot_count += count;
    }

    const uint32_t subobject_range_count = type_layout->getSubObjectRangeCount();
    subobject_ranges.resize(subobject_range_count);
    for (uint32_t subobject_range_index = 0; subobject_range_index < subobject_range_count;
         subobject_range_index++) {
        const uint32_t binding_range_index =
            type_layout->getSubObjectRangeBindingRangeIndex(subobject_range_index);
        assert(binding_ranges[binding_range_index].subobject_range_index == -1 &&
               "multiple subobject ranges map to the same binding range which is not supported.");
        binding_ranges[binding_range_index].subobject_range_index =
            static_cast<int32_t>(subobject_range_index);

        uint32_t descriptor_slot_offset = 0;
        if (auto* range_var_layout = type_layout->getSubObjectRangeOffset(subobject_range_index)) {
            descriptor_slot_offset = static_cast<uint32_t>(
                range_var_layout->getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
        }

        const slang::BindingType binding_type =
            type_layout->getBindingRangeType(binding_range_index);
        ShaderObjectLayoutHandle container_layout;
        if (binding_type == slang::BindingType::ParameterBlock ||
            binding_type == slang::BindingType::ConstantBuffer) {
            auto* leaf_type_layout =
                type_layout->getBindingRangeLeafTypeLayout(binding_range_index);
            assert(leaf_type_layout);
            container_layout = program->get_or_create_object_layout(context, leaf_type_layout);
        }

        subobject_ranges[subobject_range_index] =
            SubobjectRangeInfo{binding_range_index, descriptor_slot_offset, container_layout};
    }
}

const DescriptorSetLayoutHandle& ShaderObjectLayout::get_descriptor_set_layout() {
    assert(is_parameter_block());
    assert(has_bindings());

    if (!descriptor_set_layout) {
        DescriptorSetLayoutBuilder builder;
        if (uniform_size > 0) {
            builder.add_binding_uniform_buffer(1, vk::ShaderStageFlagBits::eAll,
                                               uniform_buffer_binding);
        }
        element_layout->add_descriptor_bindings(builder, element_binding_offset);
        descriptor_set_layout = builder.build_layout(context);
    }
    return descriptor_set_layout;
}

void ShaderObjectLayout::add_descriptor_bindings(DescriptorSetLayoutBuilder& builder,
                                                 const uint32_t binding_offset) const {
    assert(is_struct());

    auto* program_layout = program->get_program_reflection();

    // Iterate binding ranges (instead of descriptor ranges directly) to access the leaf
    // variable for resolving link-time constant array sizes.
    const uint32_t binding_range_count = type_layout->getBindingRangeCount();
    for (uint32_t binding_range_index = 0; binding_range_index < binding_range_count;
         binding_range_index++) {
        const SlangInt first_descriptor_range =
            type_layout->getBindingRangeFirstDescriptorRangeIndex(binding_range_index);
        const SlangInt descriptor_range_count =
            type_layout->getBindingRangeDescriptorRangeCount(binding_range_index);
        if (first_descriptor_range < 0 || descriptor_range_count <= 0) {
            continue;
        }

        for (SlangInt descriptor_range = first_descriptor_range;
             descriptor_range < first_descriptor_range + descriptor_range_count;
             descriptor_range++) {
            const slang::BindingType binding_type =
                type_layout->getDescriptorSetDescriptorRangeType(0, descriptor_range);

            // PushConstants are not descriptors.
            if (binding_type == slang::BindingType::PushConstant) {
                continue;
            }

            const uint32_t count = resolve_descriptor_count(
                type_layout, binding_range_index,
                type_layout->getDescriptorSetDescriptorRangeDescriptorCount(0, descriptor_range),
                program_layout);
            const uint32_t binding = static_cast<uint32_t>(
                type_layout->getDescriptorSetDescriptorRangeIndexOffset(0, descriptor_range));

            builder.add_binding(vk::DescriptorSetLayoutBinding{
                binding_offset + binding, map_slang_to_vk_descriptor_type(binding_type), count,
                vk::ShaderStageFlagBits::eAll});
        }
    }
}

std::string format_as(const ShaderObjectLayout& shader_object_layout, const std::string& indent) {
    std::string out;

    const char* kind_name = "struct";
    if (shader_object_layout.is_parameter_block()) {
        kind_name = "ParameterBlock";
    } else if (shader_object_layout.is_constant_buffer()) {
        kind_name = "ConstantBuffer";
    }
    out += fmt::format("{}kind: {}, uniform_size: {}\n", indent, kind_name,
                       shader_object_layout.get_uniform_size());

    if (shader_object_layout.is_container()) {
        out += fmt::format("{}element_binding_offset: {}, uniform_buffer_binding: {}\n", indent,
                           shader_object_layout.get_element_binding_offset(),
                           shader_object_layout.get_uniform_size() > 0
                               ? std::to_string(shader_object_layout.get_uniform_buffer_binding())
                               : "-");
        out += fmt::format("{}element:\n", indent);
        out += format_as(*shader_object_layout.get_element_layout(), indent + "  ");
        return out;
    }

    out += fmt::format("{}type layout:\n", indent);
    out += format_type_layout(shader_object_layout.get_type_layout(), 0, indent + "  ");

    out += fmt::format("{}subobjects: subobject_range_count={}\n", indent,
                       shader_object_layout.get_subobject_range_count());
    for (uint32_t subobject_range_index = 0;
         subobject_range_index < shader_object_layout.get_subobject_range_count();
         subobject_range_index++) {
        const auto& subobject_range_info =
            shader_object_layout.get_subobject_range_info(subobject_range_index);
        out += fmt::format("{}  {:>2}: binding_range_index={}, descriptor_slot_offset={}\n", indent,
                           subobject_range_index, subobject_range_info.binding_range_index,
                           subobject_range_info.descriptor_slot_offset);
        if (subobject_range_info.container_layout) {
            out += format_as(*subobject_range_info.container_layout, indent + "    ");
        }
    }

    return out;
}

} // namespace merian
