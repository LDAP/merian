#pragma once

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_module.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_session.hpp"

#include "slang-com-ptr.h"
#include "slang.h"

#include <string>
#include <vector>

namespace merian {

class SlangProgram;
using SlangProgramHandle = std::shared_ptr<SlangProgram>;

/**
 * @brief      Represents a slang program with all its entry points. This is created from a slang
 * composition that is fully linked and all dependencies are satisfied. In Vulkan this compiles to a
 * SPIRV shader module.
 */
class SlangProgram : public std::enable_shared_from_this<SlangProgram> {
  protected:
    SlangProgram(const ShaderCompileContextHandle& compile_context,
                 const SlangCompositionHandle& composition);

  public:
    ShaderModuleHandle get_shader_module(const ContextHandle& context);

    Slang::ComPtr<slang::IBlob> get_binary();

    slang::ProgramLayout* get_program_reflection() const;

    const Slang::ComPtr<slang::IComponentType>& get_program() const;

    uint64_t get_entry_point_index(const std::string& entry_point_name) const;

    const SlangCompositionHandle& get_composition();

    // ---------------------------------------------------------------
    // Global parameter discovery

    /**
     * @brief Number of global parameters.
     */
    uint32_t get_global_parameter_count() const;

    /**
     * @brief Get a global parameter's variable layout by index.
     */
    slang::VariableLayoutReflection* get_global_parameter(uint32_t index) const;

    /**
     * @brief Get a global parameter's variable layout by name. Returns nullptr if not found.
     */
    slang::VariableLayoutReflection* find_global_parameter(const std::string& name) const;

    /**
     * @brief Get names of all global parameters.
     */
    std::vector<std::string> get_global_parameter_names() const;

    /**
     * @brief Check if a global parameter exists.
     */
    bool has_global_parameter(const std::string& name) const;

    // ---------------------------------------------------------------
    // Debug

    // Print full reflection: entry points, global params, type layouts
    std::string format_reflection() const;

  public:
    static SlangProgramHandle create(const ShaderCompileContextHandle& compile_context,
                                     const SlangCompositionHandle& composition);

    // creates a program from a module.
    static SlangProgramHandle create(const ShaderCompileContextHandle& compile_context,
                                     const std::filesystem::path& path,
                                     const bool with_entry_points = true);

  private:
    const ShaderCompileContextHandle compile_context;
    const SlangCompositionHandle composition;

    SlangSessionHandle session;
    Slang::ComPtr<slang::IComponentType> program; // linked composition

    // lazyly compiled
    Slang::ComPtr<slang::IBlob> binary;
    ShaderModuleHandle shader_module{nullptr};
};

} // namespace merian
