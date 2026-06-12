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

    uint32_t get_descriptor_set_index(const std::string& param_name);

    ShaderObjectHandle create_shader_object(const ContextHandle& context,
                                            const std::string& param_name,
                                            const ResourceAllocatorHandle& allocator);

    PipelineLayoutHandle get_pipeline_layout(const ContextHandle& context);

    void bind_entry_point_parameter(const std::string& param_name,
                                    const ShaderObjectHandle& object,
                                    const CommandBufferHandle& cmd,
                                    const PipelineHandle& pipeline,
                                    const ShaderObjectAllocatorHandle& obj_allocator);

    // ---------------------------------------------------------------
    // Global parameter support (the global scope is one ParameterBlock-like container)

    bool has_globals(const ContextHandle& context);

    ShaderObjectHandle create_global_shader_object(const ContextHandle& context,
                                                   const ResourceAllocatorHandle& allocator);

    void bind_global_parameter(const ShaderObjectHandle& globals,
                               const CommandBufferHandle& cmd,
                               const PipelineHandle& pipeline,
                               const ShaderObjectAllocatorHandle& obj_allocator);

    DescriptorSetLayoutHandle get_global_set_layout(uint32_t set_index) const;

    uint32_t get_global_set_count() const;

    vk::DeviceSize get_push_constant_size() const;

    // ---------------------------------------------------------------
    // Debug

    std::string format_reflection(const ContextHandle& context);

    // ---------------------------------------------------------------
    // Parameter discovery

    bool has_parameter(const std::string& param_name) const;

    std::vector<std::string> get_parameter_block_names() const;

    std::vector<std::string> get_all_parameter_names() const;

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

    void bind_nested_parameter_blocks(const ShaderObjectHandle& object,
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
    uint32_t global_set_index = UINT32_MAX;
    std::vector<NestedParameterBlockInfo> global_nested_parameter_block_infos;
    vk::DeviceSize push_constant_size = 0;
};

} // namespace merian
