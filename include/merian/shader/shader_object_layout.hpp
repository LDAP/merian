#pragma once

#include "merian/shader/slang_program.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include "slang.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace merian {

class ShaderObjectLayout;
using ShaderObjectLayoutHandle = std::shared_ptr<ShaderObjectLayout>;

/**
 * @brief Describes the layout of a shader object (a Slang struct).
 *
 * The layout is always "relative" to an outer object, i.e. this struct might be a value,
 * ConstantBuffer or ParameterBlock member of the outer struct. This allows the same ShaderObject to
 * be nested in other objects at arbitrary position. However, this also means when writing
 * descriptors or values the writes must be forwareded with the correct offsets. Depending on the
 * nesting method, ShaderOffsets and BindingOffsets must be computed differently.
 *
 * ShaderOffsets are simply added for value nesting. For ConstantBuffer nesting, the uniform offset
 * needs to be reset, for ParameterBlock nestings uniform and binding offsets need to be reset.
 *
 * Doing it like so requires any type layout that describes this object, however, this also means
 * when binding (to a command buffer) we need to use the type layout of the concrete shader to
 * compute binding offsets to assign the object to the correct set.
 *
 * Note that ParameterCategory::DescriptorTableSlot maps to a binding in Vulkan and
 * ParameterCategory::SubElementRegisterSpace to a set.
 */
class ShaderObjectLayout {
  public:
    // Vulkan binding index for the ordinary data uniform buffer (0 when present)
    static constexpr uint32_t ORDINARY_DATA_BUFFER_BINDING = 0;

    // Pre-computed info for each sub-object range (ConstantBuffer or ParameterBlock field).
    struct SubobjectRangeInfo {
        uint32_t binding_range_index;            // index into binding_info_cache
        slang::BindingType binding_type;         // ConstantBuffer or ParameterBlock
        ShaderObjectLayoutHandle element_layout; // layout for the element type T inside
                                                 // ConstantBuffer<T>/ParameterBlock<T>
    };

    // Binding information extracted from Slang reflection
    struct BindingRangeInfo {
        uint32_t binding;                   // Vulkan binding number
        slang::BindingType type;            // Slang binding type
        uint32_t count;                     // Descriptor count
        int32_t subobject_range_index = -1; // Subobject range index or -1 if none.
    };

    // -------------------------------------

    ShaderObjectLayout(const ContextHandle& context,
                       slang::TypeLayoutReflection* type_layout,
                       const SlangProgramHandle& program);

    // -------------------------------------

    bool has_descriptor_set() const {
        return descriptor_set_layout != nullptr;
    }

    const DescriptorSetLayoutHandle& get_descriptor_set_layout() const {
        assert(has_descriptor_set());
        return descriptor_set_layout;
    }

    vk::DeviceSize get_uniform_size() const {
        return uniform_size;
    }

    bool has_ordinary_data_buffer() const {
        return uniform_size != 0;
    }

    // -------------------------------------

    slang::TypeLayoutReflection* get_type_layout() const {
        return type_layout;
    }

    const SlangProgramHandle& get_program() const {
        return program;
    }

    // -------------------------------------

    const BindingRangeInfo& get_binding_range_info(uint32_t binding_range_index) const {
        assert(binding_range_index < binding_ranges.size());
        return binding_ranges[binding_range_index];
    }

    // Returns the binding_range_index for a subobject_range_index. Returns -1 if none.
    int32_t find_subobject_range_index(uint32_t binding_range_index) const {
        assert(binding_range_index < binding_ranges.size());
        return binding_ranges[binding_range_index].subobject_range_index;
    }

    uint32_t get_subobject_range_count() const {
        return subobject_ranges.size();
    }

    const SubobjectRangeInfo& get_subobject_range_info(uint32_t index) const {
        assert(index < get_subobject_range_count());
        return subobject_ranges[index];
    }

  private:
    slang::TypeLayoutReflection* type_layout;
    SlangProgramHandle program; // keeps session alive so type_layout pointers remain valid

    DescriptorSetLayoutHandle descriptor_set_layout = nullptr;
    vk::DeviceSize uniform_size = 0;

    // Precomputed binding info for each binding range
    std::vector<BindingRangeInfo> binding_ranges;

    // Pre-computed sub-object range info (one per CB/PB field)
    std::vector<SubobjectRangeInfo> subobject_ranges;
};

std::string format_as(const ShaderObjectLayout& shader_object_layout,
                      const std::string& indent = "");

} // namespace merian
