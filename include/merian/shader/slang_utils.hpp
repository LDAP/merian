#pragma once

#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include "slang.h"

namespace merian {

// Offset within shader parameter space
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

// Binding information extracted from Slang reflection
struct BindingInfo {
    uint32_t binding;        // Vulkan binding number
    slang::BindingType type; // Slang binding type
    uint32_t count;          // Descriptor count
};

/**
 * @brief Create a Vulkan descriptor set layout from Slang type reflection.
 *
 * @param context Vulkan context
 * @param type_layout Slang type layout
 * @return Descriptor set layout handle
 */
DescriptorSetLayoutHandle
create_descriptor_set_layout_from_slang_type_layout(const ContextHandle& context,
                                                    slang::TypeLayoutReflection* type_layout);

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

// Format a ShaderOffset for debug output
std::string format_shader_offset(const ShaderOffset& offset);

/**
 * @brief Recursively format a Slang TypeLayoutReflection for debug output.
 *
 * Prints: type name/kind, uniform size, fields (with offsets and binding ranges),
 * binding ranges, descriptor set ranges, and sub-object ranges.
 * Recurses into struct/CB/PB fields up to max_depth.
 *
 * @param type_layout The type layout to format
 * @param indent Indentation prefix for each line
 * @param max_depth Maximum recursion depth (0 = this level only)
 */
std::string format_type_layout(slang::TypeLayoutReflection* type_layout,
                               const std::string& indent = "",
                               uint32_t max_depth = 2);

} // namespace merian
