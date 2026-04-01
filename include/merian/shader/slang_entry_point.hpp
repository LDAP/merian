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

    /**
     * @brief Get cached SlangObjectLayout for a named ParameterBlock parameter.
     */
    ShaderObjectLayoutHandle get_object_layout(const ContextHandle& context,
                                               const std::string& param_name);

    /**
     * @brief Get descriptor set index for a named ParameterBlock parameter.
     */
    uint32_t get_descriptor_set_index(const std::string& param_name);

    /**
     * @brief Create a ShaderObject for a named ParameterBlock parameter.
     */
    ShaderObjectHandle create_shader_object(const ContextHandle& context,
                                            const std::string& param_name,
                                            const ShaderObjectAllocatorHandle& allocator);

    /**
     * @brief Build and cache a pipeline layout from program reflection.
     *
     * Includes global parameter descriptor sets, push constants, and entry point
     * ParameterBlock descriptor sets. Global sets come first, then PB sets.
     */
    PipelineLayoutHandle get_pipeline_layout(const ContextHandle& context);

    /**
     * @brief Bind a named ParameterBlock parameter and all its nested PB sub-objects.
     */
    void bind_entry_point_parameter(const std::string& param_name,
                                    const ShaderObjectHandle& object,
                                    const CommandBufferHandle& cmd,
                                    const PipelineHandle& pipeline);

    // ---------------------------------------------------------------
    // Global parameter support

    /**
     * @brief Check if the program has global parameters that need a descriptor set.
     */
    bool has_globals(const ContextHandle& context);

    /**
     * @brief Create a ShaderObject for the global parameter scope (set 0 of globals).
     *
     * Use the returned object's cursor to set global resources and uniform data.
     * For multi-set globals (explicit [vk::binding(b, s)] across sets), this only
     * manages set 0. Additional sets must be managed manually.
     */
    ShaderObjectHandle create_global_shader_object(const ContextHandle& context,
                                                   const ShaderObjectAllocatorHandle& allocator);

    /**
     * @brief Bind the global ShaderObject at its descriptor set index.
     */
    void bind_global_parameter(const ShaderObjectHandle& globals,
                               const CommandBufferHandle& cmd,
                               const PipelineHandle& pipeline);

    /**
     * @brief Get the descriptor set layout for a specific global descriptor set.
     *
     * Useful for manually allocating descriptor sets for multi-set globals.
     * Returns nullptr if the set index is out of range.
     */
    DescriptorSetLayoutHandle get_global_set_layout(uint32_t set_index) const;

    /**
     * @brief Get the number of global descriptor sets.
     */
    uint32_t get_global_set_count() const;

    /**
     * @brief Get the push constant size (bytes) derived from reflection, or 0 if none.
     */
    vk::DeviceSize get_push_constant_size() const;

    // ---------------------------------------------------------------
    // Debug

    // Print full reflection: entry point params, pipeline layout, nested PB tree
    std::string format_reflection(const ContextHandle& context);

    // ---------------------------------------------------------------
    // Parameter discovery

    /**
     * @brief Check if a named ParameterBlock parameter exists in this entry point.
     */
    bool has_parameter(const std::string& param_name) const;

    /**
     * @brief Get names of all ParameterBlock parameters in this entry point.
     */
    std::vector<std::string> get_parameter_block_names() const;

    /**
     * @brief Get names of all entry point parameters (including non-PB like system values).
     */
    std::vector<std::string> get_all_parameter_names() const;

  public:
    static SlangProgramEntryPointHandle create(const SlangProgramHandle& program,
                                               const uint64_t entry_point_index = 0);

    static SlangProgramEntryPointHandle create(const SlangProgramHandle& program,
                                               const std::string& entry_point_name = "main");

    static SlangProgramEntryPointHandle create(const ShaderCompileContextHandle& compile_context,
                                               const std::filesystem::path& module_path,
                                               const std::string& entry_point_name = "main");

  public:
    // Tree of nested PB info: sub-object range index in parent → Vulkan set index + children
    struct NestedPBInfo {
        uint32_t subobject_range_index;
        uint32_t set_index;
        std::vector<NestedPBInfo> children; // deeply nested PBs within this PB
    };

  private:
    struct ParameterBlockInfo {
        ShaderObjectLayoutHandle object_layout;
        uint32_t descriptor_set_index;
        std::vector<NestedPBInfo>
            nested_pb_infos; // nested PBs discovered during layout construction
    };

    ParameterBlockInfo& find_or_create_param_info(const ContextHandle& context,
                                                  const std::string& param_name);

    void bind_nested_pbs(const ShaderObjectHandle& object,
                         const std::vector<NestedPBInfo>& nested_infos,
                         const CommandBufferHandle& cmd,
                         const PipelineHandle& pipeline);

    const SlangProgramHandle program;
    const uint64_t entry_point_index;

    // Cached per-parameter info
    std::unordered_map<std::string, ParameterBlockInfo> param_cache;
    PipelineLayoutHandle cached_pipeline_layout;

    // Global parameter support
    ShaderObjectLayoutHandle global_object_layout;
    uint32_t global_set_index = UINT32_MAX;
    std::vector<DescriptorSetLayoutHandle> global_set_layouts;
    std::vector<NestedPBInfo> global_nested_pb_infos;
    vk::DeviceSize push_constant_size = 0;
};

} // namespace merian
