#pragma once

#include "merian/shader/entry_point.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/shader_object_layout.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"

#include "slang.h"

#include <string>
#include <unordered_map>

namespace merian {

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;

class SlangProgramEntryPoint;
using SlangProgramEntryPointHandle = std::shared_ptr<SlangProgramEntryPoint>;

// An immutable view of one program entry point with binding helpers. Use create() for one that
// tracks a changing program.
class SlangProgramEntryPoint : public EntryPoint {

  protected:
    SlangProgramEntryPoint(const SlangProgramHandle& program, const uint64_t entry_point_index);

  public:
    virtual const char* get_name() const override;

    virtual vk::ShaderStageFlagBits get_stage() const override;

    virtual ShaderModuleHandle vulkan_shader_module(const ContextHandle& context) const override;

    slang::EntryPointReflection* get_entry_point_reflection() const;

    const SlangProgramHandle& get_program() const;

    // ---------------------------------------------------------------
    // ShaderObject helpers

    // The ParameterBlock container layout for a named entry-point parameter.
    ShaderObjectLayoutHandle get_object_layout(const ContextHandle& context,
                                               const std::string& param_name);

    // Creates the ParameterBlock object for a named entry-point parameter.
    // Create-by-type counterpart: SlangProgram::create_shader_object_for_type.
    ShaderObjectHandle create_shader_object_for_parameter(const ContextHandle& context,
                                                          const std::string& param_name,
                                                          const ResourceAllocatorHandle& allocator);

    PipelineLayoutHandle get_pipeline_layout(const ContextHandle& context);

    // Binds a shader object to the named entry-point parameter (with its nested ParameterBlocks).
    void bind(const std::string& param_name,
              const ShaderObjectHandle& object,
              const CommandBufferHandle& cmd,
              const PipelineHandle& pipeline,
              const ShaderObjectAllocatorHandle& obj_allocator);

    // ---------------------------------------------------------------
    // Global parameter support (the global scope is one ParameterBlock-like container)

    // True if the global scope owns a Vulkan descriptor set (has direct bindings).
    bool has_global_descriptor_set(const ContextHandle& context);

    ShaderObjectHandle create_global_shader_object(const ContextHandle& context,
                                                   const ResourceAllocatorHandle& allocator);

    // Binds the whole global scope object (populate it via its cursor first).
    void bind_global(const ShaderObjectHandle& globals,
                     const CommandBufferHandle& cmd,
                     const PipelineHandle& pipeline,
                     const ShaderObjectAllocatorHandle& obj_allocator);

    uint32_t get_global_set_count() const;

    vk::DeviceSize get_push_constant_size() const;

    // ---------------------------------------------------------------
    // Debug

    std::string format_reflection(const ContextHandle& context);

    // ---------------------------------------------------------------
    // Parameter reflection (entry-point parameters).
    // Global-parameter counterpart: identically-named SlangProgram::get_parameter* / has_parameter.

    uint32_t get_parameter_count() const;

    slang::VariableLayoutReflection* get_parameter(uint32_t index) const;

    // nullptr if no entry-point parameter has this name.
    slang::VariableLayoutReflection* get_parameter(const std::string& param_name) const;

    bool has_parameter(const std::string& param_name) const;

    std::vector<std::string> get_parameter_names() const;

  public:
    // An entry point that is rebuilt whenever its program changes.
    static Versioned<SlangProgramEntryPoint> create(const Versioned<SlangProgram>& program,
                                                    const uint64_t entry_point_index = 0);

    static Versioned<SlangProgramEntryPoint> create(const Versioned<SlangProgram>& program,
                                                    const std::string& entry_point_name = "main");

    static Versioned<SlangProgramEntryPoint>
    create(const ShaderCompileContextHandle& compile_context,
           const std::filesystem::path& module_path,
           const std::string& entry_point_name = "main");

  public:
    // set_index sentinel for a ParameterBlock that owns no descriptor set (no bindings).
    static constexpr uint32_t NO_DESCRIPTOR_SET = UINT32_MAX;

    // Tree of nested ParameterBlocks: sub-object range index in parent → Vulkan set index +
    // children
    struct NestedParameterBlockInfo {
        uint32_t subobject_range_index;
        uint32_t set_index;
        std::vector<NestedParameterBlockInfo> children;
    };

  private:
    struct ParameterBlockInfo {
        ShaderObjectLayoutHandle object_layout; // ParameterBlock container layout
        uint32_t descriptor_set_index;
        std::vector<NestedParameterBlockInfo> nested_parameter_block_infos;
    };

    ParameterBlockInfo& find_or_create_param_info(const ContextHandle& context,
                                                  const std::string& param_name);

    void bind_parameter_blocks(const ShaderObjectHandle& object,
                               uint32_t set_index,
                               const std::vector<NestedParameterBlockInfo>& nested_infos,
                               const CommandBufferHandle& cmd,
                               const PipelineHandle& pipeline,
                               const ShaderObjectAllocatorHandle& obj_allocator);

    const SlangProgramHandle program;
    const uint64_t entry_point_index;

    // Cached per-parameter info
    std::unordered_map<std::string, ParameterBlockInfo> param_cache;
    PipelineLayoutHandle cached_pipeline_layout;

    // Global parameter support
    ShaderObjectLayoutHandle global_object_layout;
    uint32_t global_set_index = NO_DESCRIPTOR_SET;
    std::vector<NestedParameterBlockInfo> global_nested_parameter_block_infos;
    vk::DeviceSize push_constant_size = 0;
};

} // namespace merian
