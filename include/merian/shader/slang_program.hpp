#pragma once

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_module.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_session.hpp"
#include "merian/utils/versioned.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include "slang-com-ptr.h"
#include "slang.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace merian {

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;

class ShaderObjectLayout;
using ShaderObjectLayoutHandle = std::shared_ptr<ShaderObjectLayout>;

class SlangProgram;
// Holding a handle keeps the program's session and reflection pointers alive.
using SlangProgramHandle = std::shared_ptr<SlangProgram>;

// An immutable composed and linked slang program. Use create() for one that tracks a composition.
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

    const SlangCompositionHandle& get_composition() const;

    const SlangSessionHandle& get_session() const {
        return session;
    }

    slang::TypeLayoutReflection* get_type_layout(const std::string& type_name) const;

    // Layouts are deduplicated per program: link-time constants (extern static const array
    // sizes) make binding layouts program-dependent, so sharing across programs is not safe.
    ShaderObjectLayoutHandle get_or_create_object_layout(const ContextHandle& context,
                                                         slang::TypeLayoutReflection* type_layout);

    // Creates a ParameterBlock<type_name> shader object from a named Slang type.
    // Create-for-parameter counterpart: SlangProgramEntryPoint::create_shader_object_for_parameter.
    ShaderObjectHandle create_shader_object_for_type(const ContextHandle& context,
                                                     const std::string& type_name,
                                                     const ResourceAllocatorHandle& allocator);

    // ---------------------------------------------------------------
    // Parameter reflection (global parameters).
    // Entry-point counterpart: identically-named SlangProgramEntryPoint::get_parameter* /
    // has_parameter.

    uint32_t get_parameter_count() const;

    slang::VariableLayoutReflection* get_parameter(uint32_t index) const;

    // nullptr if no global parameter has this name.
    slang::VariableLayoutReflection* get_parameter(const std::string& name) const;

    bool has_parameter(const std::string& name) const;

    std::vector<std::string> get_parameter_names() const;

    // ---------------------------------------------------------------
    // Debug

    std::string format_reflection() const;

  public:
    // recompiles when the composition changes
    static Versioned<SlangProgram> create(const ShaderCompileContextHandle& compile_context,
                                          const SlangCompositionHandle& composition);

    static Versioned<SlangProgram> create(const ShaderCompileContextHandle& compile_context,
                                          const std::filesystem::path& path,
                                          const bool with_entry_points = true);

  private:
    const ShaderCompileContextHandle compile_context;
    const SlangCompositionHandle composition;

    SlangSessionHandle session; // pinned: keeps reflection pointers valid
    Slang::ComPtr<slang::IComponentType> program;

    mutable Slang::ComPtr<slang::IBlob> binary;
    mutable ShaderModuleHandle shader_module{nullptr};

    // Weak values: layouts hold the program, strong entries would form reference cycles.
    std::unordered_map<slang::TypeLayoutReflection*, std::weak_ptr<ShaderObjectLayout>>
        object_layout_cache;
};

} // namespace merian
