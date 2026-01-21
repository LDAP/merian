#pragma once

#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include "slang.h"

namespace merian {

// Offset within shader parameter space
struct ShaderOffset {
    std::size_t uniform_byte_offset = 0; // Offset in ordinary data buffer
    uint32_t binding_range_offset = 0;   // Slang binding range index
    uint32_t binding_array_index = 0;    // Array element within binding
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
 * @brief Get binding information from a shader offset.
 *
 * @param offset The shader offset
 * @param type_layout The type layout
 * @return Binding information (binding number, type, count)
 */
BindingInfo get_binding_info_from_offset(const ShaderOffset& offset,
                                         slang::TypeLayoutReflection* type_layout);

/**
 * @brief Map Slang binding type to Vulkan descriptor type.
 *
 * @param type Slang binding type
 * @return Vulkan descriptor type
 */
vk::DescriptorType map_slang_to_vk_descriptor_type(slang::BindingType type);

} // namespace merian
