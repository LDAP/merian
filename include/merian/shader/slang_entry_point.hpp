#pragma once

#include "merian/shader/entry_point.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_object_layout.hpp"
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
    SlangObjectLayoutHandle get_object_layout(const ContextHandle& context,
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
     * Walks all descriptor sets in the entry point, creates a DescriptorSetLayout for each.
     * Does NOT require a ShaderObjectAllocator — standalone utility.
     */
    PipelineLayoutHandle get_pipeline_layout(const ContextHandle& context);

    /**
     * @brief Bind a named ParameterBlock parameter and all its nested PB sub-objects.
     *
     * Uses reflection-derived set indices (getBindingSpace()) — handles explicit
     * [[vk::binding(b, s)]] annotations and non-sequential set indices correctly.
     */
    void bind(const std::string& param_name,
              const ShaderObjectHandle& object,
              const ShaderObjectAllocatorHandle& allocator,
              const CommandBufferHandle& cmd,
              const PipelineHandle& pipeline);

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
        uint32_t sub_object_range_index;
        uint32_t set_index;
        std::vector<NestedPBInfo> children; // deeply nested PBs within this PB
    };

  private:
    struct ParameterBlockInfo {
        SlangObjectLayoutHandle object_layout;
        uint32_t descriptor_set_index;
        std::vector<NestedPBInfo>
            nested_pb_infos; // nested PBs discovered during layout construction
    };

    ParameterBlockInfo& find_or_create_param_info(const ContextHandle& context,
                                                  const std::string& param_name);

    void bind_nested_pbs(const ShaderObjectHandle& object,
                         const std::vector<NestedPBInfo>& nested_infos,
                         const ShaderObjectAllocatorHandle& allocator,
                         const CommandBufferHandle& cmd,
                         const PipelineHandle& pipeline);

    const SlangProgramHandle program;
    const uint64_t entry_point_index;

    // Cached per-parameter info
    std::unordered_map<std::string, ParameterBlockInfo> param_cache;
    PipelineLayoutHandle cached_pipeline_layout;
};

} // namespace merian
