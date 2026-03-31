#include "merian/shader/slang_utils.hpp"

#include <map>

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
    const ContextHandle& context, slang::TypeLayoutReflection* type_layout, uint32_t set_index) {

    // Track bindings by index to avoid duplicates. CB element types (obtained via
    // getElementVarLayout()->getTypeLayout()) may report a ConstantBuffer descriptor
    // range at binding 0, overlapping with the implicit UBO for uniform data.
    std::map<uint32_t, vk::DescriptorSetLayoutBinding> binding_map;

    // If the type has uniform (ordinary) data, Slang generates a ConstantBuffer at binding 0
    // in the SPIR-V. We need to include this in the descriptor set layout.
    // Uniform data UBO only lives in set 0.
    if (set_index == 0) {
        const size_t uniform_size = type_layout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);
        if (uniform_size > 0) {
            binding_map[0] = vk::DescriptorSetLayoutBinding{
                0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr};
        }
    }

    // Iterate through the descriptor set's descriptor ranges.
    // For a ParameterBlock element type, descriptor set 0 contains all resource bindings.
    // For global type layouts, each set_index corresponds to a separate Vulkan descriptor set.
    const uint32_t desc_set_count = type_layout->getDescriptorSetCount();
    if (set_index < desc_set_count) {
        const uint32_t range_count = type_layout->getDescriptorSetDescriptorRangeCount(set_index);

        for (uint32_t r = 0; r < range_count; r++) {
            const slang::BindingType kind =
                type_layout->getDescriptorSetDescriptorRangeType(set_index, r);
            const uint32_t count =
                type_layout->getDescriptorSetDescriptorRangeDescriptorCount(set_index, r);
            const vk::DescriptorType desc_type = map_slang_to_vk_descriptor_type(kind);

            // Get the actual Vulkan binding index from Slang reflection.
            // This accounts for the ConstantBuffer at binding 0 when uniform data exists.
            const uint32_t binding =
                type_layout->getDescriptorSetDescriptorRangeIndexOffset(set_index, r);

            // The manual UBO at binding 0 takes priority over descriptor ranges
            if (!binding_map.contains(binding)) {
                binding_map[binding] = vk::DescriptorSetLayoutBinding{
                    binding, desc_type, count, vk::ShaderStageFlagBits::eAll, nullptr};
            }
        }
    }

    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(binding_map.size());
    for (const auto& [idx, b] : binding_map) {
        bindings.push_back(b);
    }

    if (bindings.empty()) {
        SPDLOG_WARN("Created descriptor set layout with no bindings for type '{}'",
                    type_layout->getName() ? type_layout->getName() : "(anonymous)");
    }

    SPDLOG_DEBUG("Creating descriptor set layout for '{}' with {} bindings",
                 type_layout->getName() ? type_layout->getName() : "(anonymous)", bindings.size());
    for (size_t i = 0; i < bindings.size(); i++) {
        SPDLOG_DEBUG("  binding[{}]: binding={}, type={}, count={}", i, bindings[i].binding,
                     vk::to_string(bindings[i].descriptorType), bindings[i].descriptorCount);
    }

    return std::make_shared<DescriptorSetLayout>(context, bindings);
}

std::string
format_type_layout(slang::TypeLayoutReflection* tl, uint32_t max_depth, const std::string& indent) {
    if (!tl)
        return indent + "(null TypeLayout)\n";

    std::string out;
    const char* name = tl->getName();
    auto kind = tl->getKind();
    size_t uniform_size = tl->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);

    out += fmt::format("{}{} '{}' (uniform_size={} bytes)\n", indent,
                       slang_type_kind_to_string(kind), name ? name : "(anonymous)", uniform_size);

    // Fields
    uint32_t field_count = tl->getFieldCount();
    if (field_count > 0) {
        out += fmt::format("{}  fields ({}):\n", indent, field_count);
        for (uint32_t f = 0; f < field_count; f++) {
            auto* fv = tl->getFieldByIndex(f);
            auto* ftl = fv->getTypeLayout();
            const char* fname = fv->getVariable()->getName();
            auto fkind = ftl->getKind();
            out += fmt::format("{}    [{}] '{}': kind={}, uniform_offset={}, "
                               "binding_range_offset={}, binding_space={}\n",
                               indent, f, fname ? fname : "?", slang_type_kind_to_string(fkind),
                               fv->getOffset(SLANG_PARAMETER_CATEGORY_UNIFORM),
                               tl->getFieldBindingRangeOffset(f), fv->getBindingSpace());

            // Recurse into compound fields
            if (max_depth > 0 && (fkind == slang::TypeReflection::Kind::Struct ||
                                  fkind == slang::TypeReflection::Kind::ConstantBuffer ||
                                  fkind == slang::TypeReflection::Kind::ParameterBlock)) {
                auto* inner = ftl;
                if (fkind == slang::TypeReflection::Kind::ConstantBuffer ||
                    fkind == slang::TypeReflection::Kind::ParameterBlock) {
                    inner = ftl->getElementTypeLayout();
                }
                if (inner) {
                    out += format_type_layout(inner, max_depth - 1, indent + "      ");
                }
            }
        }
    }

    // Binding ranges
    uint32_t br_count = tl->getBindingRangeCount();
    if (br_count > 0) {
        out += fmt::format("{}  binding_ranges ({}):\n", indent, br_count);
        for (uint32_t br = 0; br < br_count; br++) {
            auto btype = tl->getBindingRangeType(br);
            auto bcount = tl->getBindingRangeBindingCount(br);
            SlangInt first_dr = tl->getBindingRangeFirstDescriptorRangeIndex(br);
            uint32_t vk_binding = 0;
            if (btype != slang::BindingType::ParameterBlock && first_dr >= 0) {
                vk_binding = tl->getDescriptorSetDescriptorRangeIndexOffset(0, first_dr);
            }
            out += fmt::format(
                "{}    [{}] type={}, count={}, first_descriptor_range={}, vk_binding={}\n", indent,
                br, slang_binding_type_to_string(btype), bcount, first_dr, vk_binding);
        }
    }

    // Descriptor set ranges
    uint32_t ds_count = tl->getDescriptorSetCount();
    if (ds_count > 0) {
        out += fmt::format("{}  descriptor_sets ({}):\n", indent, ds_count);
        for (uint32_t ds = 0; ds < ds_count; ds++) {
            uint32_t range_count = tl->getDescriptorSetDescriptorRangeCount(ds);
            out += fmt::format("{}    set {} ({} ranges):\n", indent, ds, range_count);
            for (uint32_t r = 0; r < range_count; r++) {
                auto rtype = tl->getDescriptorSetDescriptorRangeType(ds, r);
                auto rcount = tl->getDescriptorSetDescriptorRangeDescriptorCount(ds, r);
                auto roffset = tl->getDescriptorSetDescriptorRangeIndexOffset(ds, r);
                out += fmt::format("{}      [{}] vk_binding={}, type={}, count={}\n", indent, r,
                                   roffset, slang_binding_type_to_string(rtype), rcount);
            }
        }
    }

    // Sub-object ranges
    uint32_t so_count = tl->getSubObjectRangeCount();
    if (so_count > 0) {
        out += fmt::format("{}  subobject_ranges ({}):\n", indent, so_count);
        for (uint32_t i = 0; i < so_count; i++) {
            auto br_idx = tl->getSubObjectRangeBindingRangeIndex(i);
            out += fmt::format("{}    [{}] binding_range={}\n", indent, i, br_idx);
        }
    }

    return out;
}

} // namespace merian
