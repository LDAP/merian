#pragma once

#include "merian/shader/slang_program.hpp"
#include "merian/shader/slang_utils.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include "slang.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace merian {

class SlangObjectLayout;
using SlangObjectLayoutHandle = std::shared_ptr<SlangObjectLayout>;

/**
 * @brief Cached, reflection-derived description of a Slang type's Vulkan resource needs.
 *
 * Stores the descriptor set layout and uniform data size for a Slang type layout.
 * Precomputes O(1) binding info lookups and sub-object range info.
 * Keeps the SlangProgram alive to ensure TypeLayoutReflection* pointers remain valid.
 */
class SlangObjectLayout {
  public:
    // Pre-computed info for each sub-object range (ConstantBuffer or ParameterBlock field).
    struct SubObjectRangeInfo {
        uint32_t binding_range_index;           // index into binding_info_cache
        slang::BindingType binding_type;        // ConstantBuffer or ParameterBlock
        SlangObjectLayoutHandle element_layout; // layout for the element type T inside
                                                // ConstantBuffer<T>/ParameterBlock<T>
    };

    SlangObjectLayout(const ContextHandle& context,
                      slang::TypeLayoutReflection* type_layout,
                      const SlangProgramHandle& program);

    const DescriptorSetLayoutHandle& get_descriptor_set_layout() const {
        return descriptor_set_layout;
    }

    vk::DeviceSize get_uniform_size() const {
        return uniform_size;
    }

    bool has_ordinary_data_buffer() const {
        return uniform_size > 0;
    }

    // Vulkan binding index for the ordinary data uniform buffer (0 when present)
    static constexpr uint32_t ORDINARY_DATA_BUFFER_BINDING = 0;

    slang::TypeLayoutReflection* get_type_layout() const {
        return type_layout;
    }

    const SlangProgramHandle& get_program() const {
        return program;
    }

    // lookup: binding_range_index → BindingInfo{vulkan_binding, type, count}
    const BindingInfo& get_binding_info(uint32_t binding_range_index) const {
        assert(binding_range_index < binding_info_cache.size());
        return binding_info_cache[binding_range_index];
    }

    // lookup: binding_range_index → subobject_range_index. Returns -1 if not found.
    int32_t find_subobject_range_index(uint32_t binding_range_index) const {
        auto it = binding_range_to_subobject_range.find(binding_range_index);
        if (it != binding_range_to_subobject_range.end()) {
            return static_cast<int32_t>(it->second);
        }
        return -1;
    }

    // Sub-object range access
    uint32_t get_subobject_range_count() const {
        return static_cast<uint32_t>(subobject_ranges.size());
    }

    const SubObjectRangeInfo& get_subobject_range_info(uint32_t index) const {
        assert(index < subobject_ranges.size());
        return subobject_ranges[index];
    }

    // Print full reflection info for debugging
    std::string format_reflection(const std::string& indent = "") const;

  private:
    slang::TypeLayoutReflection* type_layout;
    SlangProgramHandle program; // keeps session alive so type_layout pointers remain valid
    DescriptorSetLayoutHandle descriptor_set_layout;
    vk::DeviceSize uniform_size;

    // Precomputed binding info for each binding range
    std::vector<BindingInfo> binding_info_cache;

    // Maps binding_range_index → subobject_range_index
    std::unordered_map<uint32_t, uint32_t> binding_range_to_subobject_range;

    // Pre-computed sub-object range info (one per CB/PB field)
    std::vector<SubObjectRangeInfo> subobject_ranges;
};

} // namespace merian
