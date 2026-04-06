#include "merian/shader/slang_utils.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"

namespace merian {

vk::DescriptorType map_slang_to_vk_descriptor_type(slang::BindingType type) {
    switch (type) {
    case slang::BindingType::Texture:
        return vk::DescriptorType::eSampledImage;
    case slang::BindingType::Sampler:
        return vk::DescriptorType::eSampler;
    case slang::BindingType::CombinedTextureSampler:
        return vk::DescriptorType::eCombinedImageSampler;
    case slang::BindingType::ConstantBuffer:
        return vk::DescriptorType::eUniformBuffer;
    case slang::BindingType::RawBuffer:
    case slang::BindingType::MutableRawBuffer:
        return vk::DescriptorType::eStorageBuffer;
    case slang::BindingType::MutableTexture:
        return vk::DescriptorType::eStorageImage;
    case slang::BindingType::RayTracingAccelerationStructure:
        return vk::DescriptorType::eAccelerationStructureKHR;
    case slang::BindingType::InputRenderTarget:
        return vk::DescriptorType::eInputAttachment;
    default:
        throw std::invalid_argument{"unknown binding type."};
    }
}

const char* slang_type_kind_to_string(slang::TypeReflection::Kind kind) {
    switch (kind) {
    case slang::TypeReflection::Kind::None:
        return "None";
    case slang::TypeReflection::Kind::Struct:
        return "Struct";
    case slang::TypeReflection::Kind::Array:
        return "Array";
    case slang::TypeReflection::Kind::Matrix:
        return "Matrix";
    case slang::TypeReflection::Kind::Vector:
        return "Vector";
    case slang::TypeReflection::Kind::Scalar:
        return "Scalar";
    case slang::TypeReflection::Kind::ConstantBuffer:
        return "ConstantBuffer";
    case slang::TypeReflection::Kind::Resource:
        return "Resource";
    case slang::TypeReflection::Kind::SamplerState:
        return "SamplerState";
    case slang::TypeReflection::Kind::TextureBuffer:
        return "TextureBuffer";
    case slang::TypeReflection::Kind::ShaderStorageBuffer:
        return "ShaderStorageBuffer";
    case slang::TypeReflection::Kind::ParameterBlock:
        return "ParameterBlock";
    case slang::TypeReflection::Kind::GenericTypeParameter:
        return "GenericTypeParameter";
    case slang::TypeReflection::Kind::Interface:
        return "Interface";
    case slang::TypeReflection::Kind::OutputStream:
        return "OutputStream";
    case slang::TypeReflection::Kind::Specialized:
        return "Specialized";
    case slang::TypeReflection::Kind::Feedback:
        return "Feedback";
    case slang::TypeReflection::Kind::Pointer:
        return "Pointer";
    case slang::TypeReflection::Kind::DynamicResource:
        return "DynamicResource";
    case slang::TypeReflection::Kind::MeshOutput:
        return "MeshOutput";
    case slang::TypeReflection::Kind::Enum:
        return "Enum";
    default:
        return "Unknown";
    }
}

const char* slang_binding_type_to_string(slang::BindingType type) {
    switch (type) {
    case slang::BindingType::Unknown:
        return "Unknown";
    case slang::BindingType::Sampler:
        return "Sampler";
    case slang::BindingType::Texture:
        return "Texture";
    case slang::BindingType::ConstantBuffer:
        return "ConstantBuffer";
    case slang::BindingType::ParameterBlock:
        return "ParameterBlock";
    case slang::BindingType::TypedBuffer:
        return "TypedBuffer";
    case slang::BindingType::RawBuffer:
        return "RawBuffer";
    case slang::BindingType::CombinedTextureSampler:
        return "CombinedTextureSampler";
    case slang::BindingType::InputRenderTarget:
        return "InputRenderTarget";
    case slang::BindingType::InlineUniformData:
        return "InlineUniformData";
    case slang::BindingType::RayTracingAccelerationStructure:
        return "AccelerationStructure";
    case slang::BindingType::VaryingInput:
        return "VaryingInput";
    case slang::BindingType::VaryingOutput:
        return "VaryingOutput";
    case slang::BindingType::ExistentialValue:
        return "ExistentialValue";
    case slang::BindingType::PushConstant:
        return "PushConstant";
    case slang::BindingType::MutableTexture:
        return "MutableTexture";
    case slang::BindingType::MutableTypedBuffer:
        return "MutableTypedBuffer";
    case slang::BindingType::MutableRawBuffer:
        return "MutableRawBuffer";
    default:
        return "Unknown";
    }
}

DescriptorSetLayoutHandle create_descriptor_set_layout_from_slang_type_layout(
    const ContextHandle& context,
    slang::TypeLayoutReflection* type_layout,
    slang::ProgramLayout* program_layout,
    uint32_t set_index) {

    SPDLOG_DEBUG(format_type_layout(type_layout));

    DescriptorSetLayoutBuilder builder;

    if (type_layout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM) != 0) {
        builder.add_binding_uniform_buffer(1, vk::ShaderStageFlagBits::eAll, 0);
    }

    const uint32_t desc_set_count = type_layout->getDescriptorSetCount();
    if (desc_set_count == 0) {
        return builder.build_layout(context);
    }

    assert(set_index < desc_set_count);

    // Iterate binding ranges instead of descriptor ranges so we can access the leaf variable
    // to resolve link-time constant array sizes via getElementCount(program_layout).
    const uint32_t binding_range_count = type_layout->getBindingRangeCount();
    for (uint32_t br = 0; br < binding_range_count; br++) {
        const SlangInt first_dr = type_layout->getBindingRangeFirstDescriptorRangeIndex(br);
        const SlangInt dr_count = type_layout->getBindingRangeDescriptorRangeCount(br);
        if (first_dr < 0 || dr_count <= 0)
            continue;

        for (SlangInt dr = first_dr; dr < first_dr + dr_count; dr++) {
            const slang::BindingType binding_type =
                type_layout->getDescriptorSetDescriptorRangeType(set_index, dr);
            SlangInt count =
                type_layout->getDescriptorSetDescriptorRangeDescriptorCount(set_index, dr);
            const uint32_t binding =
                type_layout->getDescriptorSetDescriptorRangeIndexOffset(set_index, dr);

            if (count == static_cast<SlangInt>(SLANG_UNKNOWN_SIZE)) {
                // Array sized by a link-time constant (extern static const).
                // Resolve via the leaf variable's type and the linked program reflection.
                if (!program_layout) {
                    throw std::runtime_error(
                        "descriptor count depends on an unresolved link-time constant "
                        "(extern static const). Ensure the constant is defined in the "
                        "composition and program_layout is passed for resolution.");
                }
                auto* leaf_var = type_layout->getBindingRangeLeafVariable(br);
                assert(leaf_var);
                auto* leaf_type = leaf_var->getType();
                assert(leaf_type && leaf_type->isArray());
                size_t resolved =
                    leaf_type->getElementCount((SlangReflection*)program_layout);
                if (resolved == SLANG_UNKNOWN_SIZE || resolved == SLANG_UNBOUNDED_SIZE) {
                    throw std::runtime_error(
                        "link-time constant array size could not be resolved. "
                        "Ensure the extern static const is defined in the composition.");
                }
                count = static_cast<SlangInt>(resolved);
            }

            const vk::DescriptorType desc_type = map_slang_to_vk_descriptor_type(binding_type);
            builder.add_binding(desc_type, static_cast<uint32_t>(count),
                                vk::ShaderStageFlagBits::eAll, nullptr, binding);
        }
    }

    return builder.build_layout(context);
}

std::string format_type_layout(slang::TypeLayoutReflection* type_layout,
                               uint32_t max_depth,
                               const std::string& indent) {
    assert(type_layout);

    const char* type_name = (type_layout->getName() != nullptr) ? type_layout->getName() : "<none>";
    const size_t uniform_size = type_layout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);
    const slang::TypeReflection::Kind kind = type_layout->getKind();

    std::string out;
    out += fmt::format("{}name: {}\n", indent, type_name);
    out += fmt::format("{}uniform size: {}\n", indent, format_size(uniform_size));
    out += fmt::format("{}kind: {}\n", indent, slang_type_kind_to_string(kind));

    if (kind == slang::TypeReflection::Kind::Array || kind == slang::TypeReflection::Kind::Matrix ||
        kind == slang::TypeReflection::Kind::Vector) {
        out +=
            fmt::format("{}element count: {}, stride: {}\n", indent, type_layout->getElementCount(),
                        type_layout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM));
    }

    // Binding ranges
    uint32_t binding_range_count = type_layout->getBindingRangeCount();
    out += fmt::format("{}binding ranges: count={}\n", indent, binding_range_count);
    for (uint32_t br = 0; br < binding_range_count; br++) {
        auto br_type = type_layout->getBindingRangeType(br);
        auto br_binding_count = type_layout->getBindingRangeBindingCount(br);

        // Is this the map from CombinedTextureSampler -> Texture array, Sampler array on some
        // platforms?
        SlangInt br_first_desc_range_index =
            type_layout->getBindingRangeFirstDescriptorRangeIndex(br);
        SlangInt br_descriptor_range_count = type_layout->getBindingRangeDescriptorRangeCount(br);

        out += fmt::format("{}  {:>2}: type={}, binding count={}, first_descriptor_range_index={}, "
                           "descriptor_range_count={}\n",
                           indent, br, slang_binding_type_to_string(br_type), br_binding_count,
                           br_first_desc_range_index, br_descriptor_range_count);
    }

    // Descriptor set ranges
    uint32_t ds_count = type_layout->getDescriptorSetCount();
    out += fmt::format("{}descriptor sets: count={}\n", indent, ds_count);
    for (uint32_t ds = 0; ds < ds_count; ds++) {
        uint32_t range_count = type_layout->getDescriptorSetDescriptorRangeCount(ds);
        uint32_t space_offset = type_layout->getDescriptorSetSpaceOffset(ds);
        out += fmt::format("{}  {:>2}: space_offset/set_index={}, descriptor_range_count={}\n",
                           indent, ds, space_offset, range_count);
        for (uint32_t r = 0; r < range_count; r++) {
            auto descriptor_range_type = type_layout->getDescriptorSetDescriptorRangeType(ds, r);
            auto descriptor_count =
                type_layout->getDescriptorSetDescriptorRangeDescriptorCount(ds, r);
            auto descriptor_offset = type_layout->getDescriptorSetDescriptorRangeIndexOffset(ds, r);
            out += fmt::format("{}    {:>2}: type={}, count={}, descriptor_offset={}\n", indent, r,
                               slang_binding_type_to_string(descriptor_range_type),
                               descriptor_count, descriptor_offset);
        }
    }

    // Sub-object ranges
    uint32_t so_count = type_layout->getSubObjectRangeCount();
    out += fmt::format("{}subobject ranges: count={}\n", indent, so_count);
    for (uint32_t i = 0; i < so_count; i++) {
        auto br_idx = type_layout->getSubObjectRangeBindingRangeIndex(i);
        auto space_offset = type_layout->getSubObjectRangeSpaceOffset(i);
        auto type = type_layout->getBindingRangeType(br_idx);
        out += fmt::format("{}  {:>2}: binding_range_index={}, type={}, space_offset={}\n", indent,
                           i, br_idx, slang_binding_type_to_string(type), space_offset);
    }

    // Fields
    const uint32_t field_count = type_layout->getFieldCount();
    out += fmt::format("{}fields: count={}\n", indent, field_count);
    for (uint32_t field_index = 0; field_index < field_count; field_index++) {
        auto* field = type_layout->getFieldByIndex(field_index);
        auto* field_type = field->getTypeLayout();
        const char* field_name = field->getVariable()->getName();
        auto field_kind = field_type->getKind();

        out += fmt::format("{}  {:>2}: name={}, kind={}, uniform_offset={}, "
                           "binding_range_offset={}, binding_space={}\n",
                           indent, field_index, (field_name != nullptr) ? field_name : "<none>",
                           slang_type_kind_to_string(field_kind),
                           field->getOffset(SLANG_PARAMETER_CATEGORY_UNIFORM),
                           type_layout->getFieldBindingRangeOffset(field_index),
                           field->getBindingSpace());

        if (max_depth > 0 && (field_kind == slang::TypeReflection::Kind::Struct ||
                              field_kind == slang::TypeReflection::Kind::ConstantBuffer ||
                              field_kind == slang::TypeReflection::Kind::ParameterBlock)) {
            auto* inner = field_type;
            if (field_kind == slang::TypeReflection::Kind::ConstantBuffer ||
                field_kind == slang::TypeReflection::Kind::ParameterBlock) {
                // unpack / dereference
                inner = field_type->getElementTypeLayout();
            }
            if (inner != nullptr) {
                out += format_type_layout(inner, max_depth - 1, indent + "  ");
            }
        }
    }

    return out;
}

} // namespace merian
