#pragma once

#include "merian/vk/shader/shader_module.hpp"
#include "merian/vk/shader/slang_session.hpp"

#include "slang-com-ptr.h"
#include "slang.h"

namespace merian {

class SlangEntryPoint;
using SlangEntryPointHandle = std::shared_ptr<SlangEntryPoint>;

class SlangProgram;
using SlangProgramHandle = std::shared_ptr<SlangProgram>;

/**
 * @brief      Represents a slang program with all its entry points. In Vulkan this compiles to a
 * SPIRV shader module.
 */
class SlangProgram : public std::enable_shared_from_this<SlangProgram> {

  public:
    ShaderModuleHandle get_shader_module(const ContextHandle& context);

    slang::ProgramLayout* get_program_reflection() const;

    const Slang::ComPtr<slang::IComponentType>& get_program() const;

    SlangEntryPointHandle get_entry_point_by_index(const uint64_t entry_point_index = 0);

    SlangEntryPointHandle get_entry_point_by_name(const std::string& entry_point_name = "main");

  private:
    const SlangSessionHandle session;
    const Slang::ComPtr<slang::IComponentType> program;

    // lazyly compiled
    ShaderModuleHandle shader_module{nullptr};
};

} // namespace merian
