#pragma once

#include "merian/vk/shader/entry_point.hpp"
#include "merian/vk/shader/slang_program.hpp"

#include "slang.h"

namespace merian {

class SlangShaderModule;
using SlangShaderModuleHandle = std::shared_ptr<SlangShaderModule>;

class SlangShaderModule : public ShaderModule {};

class SlangEntryPoint;
using SlangEntryPointHandle = std::shared_ptr<SlangEntryPoint>;

class SlangEntryPoint : public EntryPoint {

  protected:
    SlangEntryPoint(const SlangProgramHandle& program, const uint64_t entry_point_index);

  public:
    virtual const char* get_name() const override;

    virtual vk::ShaderStageFlagBits get_stage() const override;

    virtual ShaderModuleHandle vulkan_shader_module(const ContextHandle& context) const override;

    slang::EntryPointReflection* get_entry_point_reflection() const;

    const SlangProgramHandle& get_program() const;

  public:
    static SlangEntryPointHandle create(const SlangProgramHandle& program,
                                        const uint64_t entry_point_index);

    static SlangEntryPointHandle create(const SlangProgramHandle& program,
                                        const std::string& entry_point_name);

  private:
    const SlangProgramHandle program;
    const uint64_t entry_point_index;
};

} // namespace merian
