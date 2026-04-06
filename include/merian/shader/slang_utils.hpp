#pragma once

#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include "slang.h"

namespace merian {

/* Offset within shader parameter space.
 *
 * Quote from shader-slang.com/docs/shader-cursors:
 * Every type layout can be broken down as zero or more bytes of ordinary data, and zero or more
 * binding ranges. All of the binding ranges of a type are grouped together for counting and
 * indexing, so that no matter how complicated of a type an application is working with a
 * ShaderCursor implementation need only track two things: a byte offset for ordinary data, and an
 * index for a binding range.
 */
struct ShaderOffset {
    std::size_t uniform_byte_offset = 0; // Offset in ordinary data buffer
    uint32_t binding_range_offset = 0;   // Slang binding range index
    uint32_t binding_array_index = 0;    // Array element within binding

    ShaderOffset operator+(const ShaderOffset& o) const {
        return {uniform_byte_offset + o.uniform_byte_offset,
                binding_range_offset + o.binding_range_offset,
                binding_array_index + o.binding_array_index};
    }

    ShaderOffset& operator+=(const ShaderOffset& o) {
        uniform_byte_offset += o.uniform_byte_offset;
        binding_range_offset += o.binding_range_offset;
        binding_array_index += o.binding_array_index;
        return *this;
    }
};

inline std::string format_as(const ShaderOffset& offset) {
    return fmt::format("(uniform_byte={}, binding_range={}, binding_array={})",
                       offset.uniform_byte_offset, offset.binding_range_offset,
                       offset.binding_array_index);
}

// Describe the locations where shader parameter are bound. Most shader parameters in Vulkan
// simply consume a single `binding` in a set, but we also need to deal with
// parameters that represent push-constant ranges and subobjects which may be bound at set offset.
struct BindingOffset {
    uint32_t set = 0;
    uint32_t binding = 0;
    uint32_t push_constant_range = 0;

    BindingOffset operator+(const BindingOffset& o) const {
        return {
            set + o.set,
            binding + o.binding,
            push_constant_range + o.push_constant_range,
        };
    }

    BindingOffset& operator+=(const BindingOffset& o) {
        set += o.set;
        binding += o.binding;
        push_constant_range += o.push_constant_range;
        return *this;
    }
};

inline std::string format_as(const BindingOffset& offset) {
    return fmt::format("(set_offset={}, binding_offset={}, push_constant_range={})", offset.set,
                       offset.binding, offset.push_constant_range);
}

/**
 * @brief Map Slang binding type to Vulkan descriptor type.
 *
 * @param type Slang binding type
 * @return Vulkan descriptor type
 */
vk::DescriptorType map_slang_to_vk_descriptor_type(slang::BindingType type);

// Convert Slang TypeReflection::Kind to string for debug output
const char* slang_type_kind_to_string(slang::TypeReflection::Kind kind);

// Convert Slang BindingType to string for debug output
const char* slang_binding_type_to_string(slang::BindingType type);

/**
 * @brief Create a Vulkan descriptor set layout from Slang type reflection, assuming that the
 * offsets are all 0, i.e. this might not result in the correct layout for nested objects.
 *
 * @param context Vulkan context
 * @param type_layout Slang type layout
 * @param set_index The (relative) descriptor set to extract, i.e. for ParameterBlock element types,
 *                  set 0 contains all bindings.
 * @return Descriptor set layout handle
 */
DescriptorSetLayoutHandle create_descriptor_set_layout_from_slang_type_layout(
    const ContextHandle& context,
    slang::TypeLayoutReflection* type_layout,
    slang::ProgramLayout* program_layout = nullptr,
    uint32_t set_index = 0);

/**
 * @brief Recursively format a Slang TypeLayoutReflection for debug output.
 *
 * Recurses into struct/ConstantBuffer/ParameterBlock fields up to max_depth.
 *
 * @param type_layout The type layout to format
 * @param indent Indentation prefix for each line
 * @param max_depth Maximum recursion depth (0 = this level only)
 */
std::string format_type_layout(slang::TypeLayoutReflection* type_layout,
                               uint32_t max_depth = 0,
                               const std::string& indent = "");

} // namespace merian
