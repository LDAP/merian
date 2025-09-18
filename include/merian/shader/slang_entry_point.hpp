#pragma once

#include "merian/shader/entry_point.hpp"
#include "merian/shader/slang_program.hpp"

#include "slang.h"

namespace merian {

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

  public:
    static SlangProgramEntryPointHandle create(const SlangProgramHandle& program,
                                               const uint64_t entry_point_index = 0);

    static SlangProgramEntryPointHandle create(const SlangProgramHandle& program,
                                               const std::string& entry_point_name = "main");

    static SlangProgramEntryPointHandle create(const ShaderCompileContextHandle& compile_context,
                                               const std::filesystem::path& module_path,
                                               const std::string& entry_point_name = "main");

  private:
    const SlangProgramHandle program;
    const uint64_t entry_point_index;
};

} // namespace merian
