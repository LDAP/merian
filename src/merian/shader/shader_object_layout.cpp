#include "merian/shader/shader_object_layout.hpp"
#include "merian/shader/slang_utils.hpp"

namespace merian {

ShaderObjectLayout::ShaderObjectLayout(const ContextHandle& context,
                                       slang::TypeLayoutReflection* type_layout,
                                       const SlangProgramHandle& program)
    : type_layout(type_layout), program(program) {
    assert(type_layout);
    assert(program);

    descriptor_set_layout =
        create_descriptor_set_layout_from_slang_type_layout(context, type_layout);

    uniform_size = type_layout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);

    // Precompute binding info cache: binding_range_index → BindingInfo
    const uint32_t binding_range_count = type_layout->getBindingRangeCount();
    binding_info_cache.resize(binding_range_count);
    for (uint32_t br = 0; br < binding_range_count; br++) {
        const slang::BindingType kind = type_layout->getBindingRangeType(br);
        const uint32_t count = type_layout->getBindingRangeBindingCount(br);

        uint32_t vulkan_binding = 0;

        // ParameterBlock binding ranges don't have descriptors in set 0
        if (kind != slang::BindingType::ParameterBlock) {
            const SlangInt first_dr = type_layout->getBindingRangeFirstDescriptorRangeIndex(br);
            if (first_dr >= 0) {
                vulkan_binding =
                    type_layout->getDescriptorSetDescriptorRangeIndexOffset(0, first_dr);
            }
        }

        binding_info_cache[br] = BindingInfo{vulkan_binding, kind, count};
    }

    // Precompute sub-object ranges: one entry per CB/PB field, with element layout.
    // Follows slang-rhi's approach: use getElementVarLayout()->getTypeLayout() to get the
    // unwrapped element type for all sub-object ranges, skip ExistentialValue.
    // Only create element layouts for ConstantBuffer and ParameterBlock types
    // (other types like RWStructuredBuffer would cause infinite recursion).
    const uint32_t sor_count = type_layout->getSubObjectRangeCount();
    subobject_ranges.resize(sor_count);
    for (uint32_t i = 0; i < sor_count; i++) {
        const uint32_t br_index = type_layout->getSubObjectRangeBindingRangeIndex(i);
        binding_range_to_subobject_range[br_index] = i;

        const slang::BindingType kind = type_layout->getBindingRangeType(br_index);
        auto* leaf_tl = type_layout->getBindingRangeLeafTypeLayout(br_index);

        ShaderObjectLayoutHandle element_layout;
        if (leaf_tl != nullptr) {
            if (kind == slang::BindingType::ParameterBlock) {
                // For PBs, use getElementTypeLayout() which preserves descriptor set context
                // (needed for create_descriptor_set_layout_from_slang_type_layout)
                auto* element_tl = leaf_tl->getElementTypeLayout();
                if (element_tl != nullptr) {
                    element_layout =
                        std::make_shared<ShaderObjectLayout>(context, element_tl, program);
                }
            } else if (kind == slang::BindingType::ConstantBuffer) {
                // For CBs, use getElementVarLayout()->getTypeLayout() for clean struct layouts
                auto* var_layout = leaf_tl->getElementVarLayout();
                if (var_layout != nullptr) {
                    auto* sub_type_layout = var_layout->getTypeLayout();
                    if (sub_type_layout != nullptr) {
                        element_layout =
                            std::make_shared<ShaderObjectLayout>(context, sub_type_layout, program);
                    }
                }
            }
        }

        subobject_ranges[i] = SubObjectRangeInfo{br_index, kind, element_layout};
    }
}

std::string format_as(const ShaderObjectLayout& shader_object_layout, const std::string& indent) {
    std::string out;
    out += fmt::format("{}type layout:\n", indent);
    out += format_type_layout(shader_object_layout.get_type_layout(), 0, indent + "  ");

    out += fmt::format("{}subobjects: subobject_range_count={}\n", indent,
                       shader_object_layout.get_subobject_range_count());
    for (uint32_t subobject_range_index = 0;
         subobject_range_index < shader_object_layout.get_subobject_range_count();
         subobject_range_index++) {
        const auto& subobject_range_info =
            shader_object_layout.get_subobject_range_info(subobject_range_index);
        out += fmt::format("{}  {:>2}: binding_range_index={}, type={}\n", indent,
                           subobject_range_index, subobject_range_info.binding_range_index,
                           slang_binding_type_to_string(subobject_range_info.binding_type));
        if (subobject_range_info.element_layout) {
            out += format_as(*subobject_range_info.element_layout, indent + "    ");
        }
    }

    return out;
}

} // namespace merian
