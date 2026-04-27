#pragma once

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_module.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_session.hpp"
#include "merian/utils/versionable.hpp"

#include "slang-com-ptr.h"
#include "slang.h"

#include <string>
#include <vector>

namespace merian {

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;
class ShaderObjectAllocator;
using ShaderObjectAllocatorHandle = std::shared_ptr<ShaderObjectAllocator>;

class SlangProgram;
using SlangProgramHandle = std::shared_ptr<SlangProgram>;

/**
 * @brief Represents a slang program with all its entry points.
 *
 * Created from a SlangComposition that is fully linked with all dependencies satisfied.
 * Compiles to a SPIR-V shader module for Vulkan.
 *
 * Implements Versionable: when the underlying composition changes, the program
 * automatically recompiles (creating a fresh SlangSession) and increments its version.
 */
class SlangProgram : public Versionable, public std::enable_shared_from_this<SlangProgram> {
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

    const SlangSessionHandle& get_session() const {
        return session;
    }

    // ---------------------------------------------------------------
    // Rebuild

    /**
     * @brief Recompile from the current composition using a fresh session.
     *
     * Creates a new SlangSession (force_new=true), relinks the composition,
     * clears cached binary/shader module, and increments version.
     */
    void rebuild();

    // ---------------------------------------------------------------
    // Type layout

    /**
     * @brief Get the type layout for a named type in this program.
     *
     * Useful for creating ShaderObjectLayouts from a type defined in the program.
     */
    slang::TypeLayoutReflection* get_type_layout(const std::string& type_name) const;

    /**
     * @brief Create a ShaderObject for a named type in this program.
     *
     * Convenience that combines get_type_layout, ShaderObjectLayout, and ShaderObject construction.
     */
    ShaderObjectHandle create_shader_object(const ContextHandle& context,
                                            const std::string& type_name,
                                            const ShaderObjectAllocatorHandle& obj_allocator);

    // ---------------------------------------------------------------
    // Global parameter discovery

    uint32_t get_global_parameter_count() const;
    slang::VariableLayoutReflection* get_global_parameter(uint32_t index) const;
    slang::VariableLayoutReflection* find_global_parameter(const std::string& name) const;
    std::vector<std::string> get_global_parameter_names() const;
    bool has_global_parameter(const std::string& name) const;

    // ---------------------------------------------------------------
    // Debug

    std::string format_reflection() const;

  public:
    static SlangProgramHandle create(const ShaderCompileContextHandle& compile_context,
                                     const SlangCompositionHandle& composition);

    static SlangProgramHandle create(const ShaderCompileContextHandle& compile_context,
                                     const std::filesystem::path& path,
                                     const bool with_entry_points = true);

  private:
    const ShaderCompileContextHandle compile_context;
    const SlangCompositionHandle composition;

    SlangSessionHandle session;
    Slang::ComPtr<slang::IComponentType> program; // linked composition

    // lazily compiled
    Slang::ComPtr<slang::IBlob> binary;
    ShaderModuleHandle shader_module{nullptr};
};

} // namespace merian
