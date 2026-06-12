#pragma once

#include "merian/shader/slang_program.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include "slang.h"

#include <memory>
#include <vector>

namespace merian {

class DescriptorSetLayoutBuilder;

class ShaderObjectLayout;
using ShaderObjectLayoutHandle = std::shared_ptr<ShaderObjectLayout>;

/**
 * @brief Layout of a shader object: a plain Slang struct or a single-element container
 * (ConstantBuffer<T> / ParameterBlock<T>) holding such a struct.
 *
 * Struct layouts are relative: uniform byte offsets and Vulkan bindings are expressed within the
 * enclosing container. Container layouts carry the offsets Slang assigns to their element
 * (getElementVarLayout / getContainerVarLayout), so offsets are never re-derived at write or bind
 * time. The global parameter scope is treated as a ParameterBlock container around the global
 * struct (Slang wraps global uniform data in an implicit ConstantBuffer the same way).
 *
 * Only ParameterBlock containers own a Vulkan descriptor set; their DescriptorSetLayout is built
 * lazily on first use and shared through the per-program layout cache
 * (SlangProgram::get_or_create_object_layout).
 */
class ShaderObjectLayout {
  public:
    enum class Kind : uint8_t {
        STRUCT,
        CONSTANT_BUFFER,
        PARAMETER_BLOCK,
    };

    // Binding information for one binding range of a struct layout.
    struct BindingRangeInfo {
        uint32_t binding;        // relative to the enclosing container's element binding offset
        slang::BindingType type; // Slang binding type
        uint32_t count;          // descriptor count (link-time array sizes resolved)
        uint32_t slot_base;      // first index into the flattened resource-slot record
        int32_t subobject_range_index = -1; // sub-object range index or -1 if none
    };

    // One ConstantBuffer or ParameterBlock field of a struct layout.
    struct SubobjectRangeInfo {
        uint32_t binding_range_index;
        // Field offset within the struct, in descriptor table slots.
        uint32_t descriptor_slot_offset;
        ShaderObjectLayoutHandle container_layout;
    };

    // -------------------------------------

    // type_layout: struct, ConstantBuffer<T> or ParameterBlock<T>. as_scope_container treats the
    // layout as a set-owning ParameterBlock regardless of its kind (the global parameter scope).
    // Use SlangProgram::get_or_create_object_layout instead of constructing directly.
    ShaderObjectLayout(const ContextHandle& context,
                       slang::TypeLayoutReflection* type_layout,
                       const SlangProgramHandle& program,
                       const bool as_scope_container = false);

    // -------------------------------------

    Kind get_kind() const {
        return kind;
    }

    bool is_struct() const {
        return kind == Kind::STRUCT;
    }

    bool is_container() const {
        return kind != Kind::STRUCT;
    }

    bool is_parameter_block() const {
        return kind == Kind::PARAMETER_BLOCK;
    }

    bool is_constant_buffer() const {
        return kind == Kind::CONSTANT_BUFFER;
    }

    // -------------------------------------
    // Container layouts

    const ShaderObjectLayoutHandle& get_element_layout() const {
        assert(is_container());
        return element_layout;
    }

    // Byte offset of the element's uniform data within the container's uniform buffer.
    vk::DeviceSize get_element_uniform_offset() const {
        assert(is_container());
        return element_uniform_offset;
    }

    // Descriptor slot offset of the element's bindings within the enclosing descriptor set.
    uint32_t get_element_binding_offset() const {
        assert(is_container());
        return element_binding_offset;
    }

    // Descriptor slot of the container's implicit uniform buffer (valid if uniform data exists).
    uint32_t get_uniform_buffer_binding() const {
        assert(is_container() && get_uniform_size() > 0);
        return uniform_buffer_binding;
    }

    // Whether a ParameterBlock contributes a Vulkan descriptor set. Answered from reflection
    // without building the DescriptorSetLayout.
    bool has_bindings() const {
        assert(is_parameter_block());
        return parameter_block_has_bindings;
    }

    // Lazily built; only valid for ParameterBlock layouts with bindings.
    const DescriptorSetLayoutHandle& get_descriptor_set_layout();

    // -------------------------------------
    // Struct layouts (containers forward to their element)

    vk::DeviceSize get_uniform_size() const {
        return uniform_size;
    }

    const BindingRangeInfo& get_binding_range_info(const uint32_t binding_range_index) const {
        assert(is_struct());
        assert(binding_range_index < binding_ranges.size());
        return binding_ranges[binding_range_index];
    }

    // Returns the subobject_range_index for a binding_range_index. Returns -1 if none.
    int32_t find_subobject_range_index(const uint32_t binding_range_index) const {
        assert(is_struct());
        assert(binding_range_index < binding_ranges.size());
        return binding_ranges[binding_range_index].subobject_range_index;
    }

    uint32_t get_subobject_range_count() const {
        const auto& ranges = is_container() ? element_layout->subobject_ranges : subobject_ranges;
        return static_cast<uint32_t>(ranges.size());
    }

    const SubobjectRangeInfo& get_subobject_range_info(const uint32_t index) const {
        const auto& ranges = is_container() ? element_layout->subobject_ranges : subobject_ranges;
        assert(index < ranges.size());
        return ranges[index];
    }

    // Total number of resource slots (for the replay record of a struct ShaderObject).
    uint32_t get_slot_count() const {
        assert(is_struct());
        return slot_count;
    }

    // -------------------------------------

    slang::TypeLayoutReflection* get_type_layout() const {
        return type_layout;
    }

    const SlangProgramHandle& get_program() const {
        return program;
    }

  private:
    void init_struct(const ContextHandle& context);
    void init_container(const ContextHandle& context);

    // Adds all descriptor ranges of a struct layout, shifted by binding_offset.
    void add_descriptor_bindings(DescriptorSetLayoutBuilder& builder,
                                 const uint32_t binding_offset) const;

  private:
    slang::TypeLayoutReflection* type_layout;
    SlangProgramHandle program; // keeps session alive so type_layout pointers remain valid
    ContextHandle context;      // for lazy DescriptorSetLayout building
    Kind kind;

    // container layouts
    ShaderObjectLayoutHandle element_layout = nullptr;
    vk::DeviceSize element_uniform_offset = 0;
    uint32_t element_binding_offset = 0;
    uint32_t uniform_buffer_binding = 0;
    bool parameter_block_has_bindings = false;
    DescriptorSetLayoutHandle descriptor_set_layout = nullptr;

    // struct layouts
    vk::DeviceSize uniform_size = 0;
    uint32_t slot_count = 0;
    std::vector<BindingRangeInfo> binding_ranges;
    std::vector<SubobjectRangeInfo> subobject_ranges;
};

std::string format_as(const ShaderObjectLayout& shader_object_layout,
                      const std::string& indent = "");

} // namespace merian
