#pragma once

#include "merian/shader/entry_point.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_compiler.hpp"
#include "merian/shader/shader_module.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_global_session.hpp"
#include "merian/utils/hash.hpp"

#include "slang-com-ptr.h"
#include "slang.h"
#include <filesystem>
#include <functional>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace merian {

class SlangProgram;
using SlangProgramHandle = std::shared_ptr<SlangProgram>;

class SlangSession;
using SlangSessionHandle = std::shared_ptr<SlangSession>;

// A wrapper around a slang session.
class SlangSession {
  protected:
    SlangSession(const ShaderCompileContextHandle& shader_compile_context)
        : weak_compile_context(shader_compile_context) {
        const auto global_session = get_global_slang_session();

        slang::SessionDesc slang_session_desc = {};

        slang::TargetDesc target_desc = {};
        switch (shader_compile_context->get_target()) {
        case MERIAN_SPIRV_VERSION_1_0:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_0");
            break;
        case MERIAN_SPIRV_VERSION_1_1:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_1");
            break;
        case MERIAN_SPIRV_VERSION_1_2:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_2");
            break;
        case MERIAN_SPIRV_VERSION_1_3:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_3");
            break;
        case MERIAN_SPIRV_VERSION_1_4:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_4");
            break;
        case MERIAN_SPIRV_VERSION_1_5:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_5");
            break;
        case MERIAN_SPIRV_VERSION_1_6:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_6");
            break;
        default:
            SPDLOG_WARN("Target not unknown!");
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_6");
            break;
        }

        slang_session_desc.targets = &target_desc;
        slang_session_desc.targetCount = 1;

        std::vector<slang::PreprocessorMacroDesc> preprocessor_macros;
        preprocessor_macros.reserve(shader_compile_context->get_preprocessor_macros().size());

        for (const auto& macro : shader_compile_context->get_preprocessor_macros()) {
            preprocessor_macros.emplace_back(macro.first.c_str(), macro.second.c_str());
        }
        slang_session_desc.preprocessorMacros = preprocessor_macros.data();
        slang_session_desc.preprocessorMacroCount = (SlangInt)preprocessor_macros.size();

        std::vector<std::string> str_search_paths;
        std::vector<const char*> search_paths;
        search_paths.reserve(
            shader_compile_context->get_search_path_file_loader().get_search_paths().size());
        str_search_paths.reserve(
            shader_compile_context->get_search_path_file_loader().get_search_paths().size());
        // conversion for Windows...
        for (const auto& search_path : shader_compile_context->get_search_path_file_loader()) {
            str_search_paths.emplace_back(search_path.string());
        }
        for (const auto& search_path : str_search_paths) {
            search_paths.emplace_back(search_path.c_str());
        }
        slang_session_desc.searchPaths = search_paths.data();
        slang_session_desc.searchPathCount = (SlangInt)search_paths.size();

        if (cache_enabled()) {
            file_system = create_file_system(module_cache_dir(*shader_compile_context),
                                             [this] { return modules_canonically_named(); });
            slang_session_desc.fileSystem = file_system.get();
        }

        std::array<slang::CompilerOptionEntry, 6> options = {
            {
                {
                    slang::CompilerOptionName::UseUpToDateBinaryModule,
                    {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
                },
                {
                    slang::CompilerOptionName::VulkanUseEntryPointName,
                    {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
                },
                {
                    slang::CompilerOptionName::EmitSpirvDirectly,
                    {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
                },
                {
                    slang::CompilerOptionName::EmitSpirvViaGLSL,
                    {slang::CompilerOptionValueKind::Int, 0, 0, nullptr, nullptr},
                },
                {
                    slang::CompilerOptionName::Optimization,
                    {slang::CompilerOptionValueKind::Int,
                     static_cast<int32_t>(shader_compile_context->get_optimization_level()), 0,
                     nullptr, nullptr},
                },
                {
                    slang::CompilerOptionName::DebugInformation,
                    {slang::CompilerOptionValueKind::Int,
                     static_cast<int32_t>(
                         debug_info_level(shader_compile_context->should_generate_debug_info())),
                     0, nullptr, nullptr},
                },
            },
        };
        slang_session_desc.compilerOptionEntries = options.data();
        slang_session_desc.compilerOptionEntryCount = options.size();
        target_desc.compilerOptionEntries = options.data();
        target_desc.compilerOptionEntryCount = options.size();

        global_session->createSession(slang_session_desc, session.writeRef());
    }

  public:
    ShaderCompileContextHandle get_compile_context() {
        return compile_context();
    }

    // The path can be used as path-based import statement the
    // module. The name is the stem (final part without its suffix) of this path. If the source path
    // should not be the same as the path for path-based includes use "source_path".
    //
    // Note: The returned module is only valid as long as this session is valid
    Slang::ComPtr<slang::IModule>
    load_module_from_path(const std::filesystem::path& path,
                          const std::optional<std::filesystem::path>& source_path = std::nullopt) {
        return load_module_from_path(path.stem().string(), path, source_path);
    }

    // The path can be used as path-based import statement the
    // module. The name is the stem (final part without its suffix) of this path. If the source path
    // should not be the same as the path for path-based includes use "source_path".
    //
    // Note: The returned module is only valid as long as this session is valid
    Slang::ComPtr<slang::IModule>
    load_module_from_path(const std::string& name,
                          const std::filesystem::path& path,
                          const std::optional<std::filesystem::path>& source_path = std::nullopt) {
        const ShaderCompileContextHandle context = compile_context();
        std::optional<std::string> source;
        if (source_path) {
            source = context->get_search_path_file_loader().find_and_load_file(source_path.value());
        } else {
            source = context->get_search_path_file_loader().find_and_load_file(path);
        }

        if (!source) {
            throw ShaderCompiler::compilation_failed(
                fmt::format("Compiling module {} from {} failed: Not found", name, path.string()));
        }

        return load_module_from_source(name, *source, path);
    }

    // Loads a module from a source string. The path can be used as path-based import statement the
    // module.
    //
    // Note: The returned module is only valid as long as this session is valid
    Slang::ComPtr<slang::IModule>
    load_module_from_source(const std::string& name,
                            const std::string& source,
                            const std::optional<std::filesystem::path>& path);

    static Slang::ComPtr<slang::IEntryPoint> find_entry_point(Slang::ComPtr<slang::IModule>& module,
                                                              const std::string& name) {
        Slang::ComPtr<slang::IEntryPoint> entry_point;
        module->findEntryPointByName(name.c_str(), entry_point.writeRef());
        return entry_point;
    }

    static Slang::ComPtr<slang::IEntryPoint>
    get_defined_entry_point(Slang::ComPtr<slang::IModule>& module, const uint32_t index = 0) {
        Slang::ComPtr<slang::IEntryPoint> entry_point;
        module->getDefinedEntryPoint((SlangInt32)index, entry_point.writeRef());
        return entry_point;
    }

    static uint32_t get_defined_entry_point_count(Slang::ComPtr<slang::IModule>& module) {
        return (uint32_t)module->getDefinedEntryPointCount();
    }

    // throws compilation failed if not found
    static Slang::ComPtr<slang::IEntryPoint>
    find_entry_point_or_fail(Slang::ComPtr<slang::IModule>& module, const std::string& name) {
        Slang::ComPtr<slang::IEntryPoint> entry_point = find_entry_point(module, name);
        if (entry_point == nullptr) {
            throw ShaderCompiler::compilation_failed(fmt::format(
                "entrypoint {} could not be found in module {}", name, module->getName()));
        }
        return entry_point;
    }

    Slang::ComPtr<slang::ITypeConformance> create_type_conformance(slang::TypeReflection* type,
                                                                   slang::TypeReflection* interface,
                                                                   int64_t& id) {
        Slang::ComPtr<slang::ITypeConformance> type_conformance;
        Slang::ComPtr<slang::IBlob> diagnostics_blob;

        SlangResult result = session->createTypeConformanceComponentType(
            type, interface, type_conformance.writeRef(), id, diagnostics_blob.writeRef());

        if (SLANG_FAILED(result)) {
            // type does not conform to interface.
            throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
        }

        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang creating type conformance failed. Diagnostics: {}",
                         diagnostics_as_string(diagnostics_blob));
        }

        return type_conformance;
    }

    Slang::ComPtr<slang::ITypeConformance>
    create_type_conformance(const Slang::ComPtr<slang::IComponentType>& type_component,
                            const std::string& type_name,
                            const Slang::ComPtr<slang::IComponentType>& interface_component,
                            const std::string& interface_type_name,
                            int64_t& id) {
        slang::TypeReflection* type =
            type_component->getLayout()->findTypeByName(type_name.c_str());
        slang::TypeReflection* interface =
            interface_component->getLayout()->findTypeByName(interface_type_name.c_str());

        if (type == nullptr) {
            throw ShaderCompiler::compilation_failed{
                fmt::format("{} not found in in type component", type_name)};
        }

        if (interface == nullptr) {
            throw ShaderCompiler::compilation_failed{
                fmt::format("{} not found in in interface component", interface_type_name)};
        }

        return create_type_conformance(type, interface, id);
    }

    // Creates a type conformance. Assumes that the type and interface are known to component.
    // "id" is the preffered id that is used for the createDynamicObject<>(id, ...) method in Slang
    // or -1 if the compiler should choose one.
    Slang::ComPtr<slang::ITypeConformance>
    create_type_conformance(const Slang::ComPtr<slang::IComponentType>& component,
                            const std::string& type_name,
                            const std::string& interface_type_name,
                            int64_t& id) {
        return create_type_conformance(component, type_name, component, interface_type_name, id);
    }

    // Compose modules, entry points and type conformances to a (linkable) component.
    Slang::ComPtr<slang::IComponentType>
    compose(const vk::ArrayProxy<slang::IComponentType*>& components) {
        Slang::ComPtr<slang::IComponentType> composed;
        Slang::ComPtr<slang::IBlob> diagnostics_blob;

        SlangResult result = session->createCompositeComponentType(
            components.data(), components.size(), composed.writeRef(), diagnostics_blob.writeRef());

        if (SLANG_FAILED(result)) {
            throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
        }

        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang composing components. Diagnostics: {}",
                         diagnostics_as_string(diagnostics_blob));
        }

        return composed;
    }

    Slang::ComPtr<slang::IComponentType>
    compose(const vk::ArrayProxy<Slang::ComPtr<slang::IComponentType>>& components) {
        Slang::ComPtr<slang::IComponentType> composed;
        Slang::ComPtr<slang::IBlob> diagnostics_blob;

        SlangResult result = session->createCompositeComponentType(
            (slang::IComponentType**)components.data(), components.size(), composed.writeRef(),
            diagnostics_blob.writeRef());

        if (SLANG_FAILED(result)) {
            throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
        }

        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang composing components. Diagnostics: {}",
                         diagnostics_as_string(diagnostics_blob));
        }

        return composed;
    }

    Slang::ComPtr<slang::IComponentType> compose(const SlangCompositionHandle& composition);

    // creates a composite of the module with all its entrypoints.
    Slang::ComPtr<slang::IComponentType>
    compose_all_entrypoints(Slang::ComPtr<slang::IModule>& module) {
        std::vector<Slang::ComPtr<slang::IComponentType>> composite(
            get_defined_entry_point_count(module) + 1);
        composite[0] = module.get();
        for (uint32_t i = 0; i < get_defined_entry_point_count(module); i++) {
            composite[i + 1] = get_defined_entry_point(module, i);
        }

        return compose(composite);
    }

    static Slang::ComPtr<slang::IComponentType>
    link(const Slang::ComPtr<slang::IComponentType>& composed_programm) {
        Slang::ComPtr<slang::IComponentType> linked;
        Slang::ComPtr<slang::IBlob> diagnostics_blob;

        SlangResult result =
            composed_programm->link(linked.writeRef(), diagnostics_blob.writeRef());

        if (SLANG_FAILED(result)) {
            throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
        }

        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang linking. Diagnostics: {}", diagnostics_as_string(diagnostics_blob));
        }

        return linked;
    }

    static Slang::ComPtr<slang::IBlob>
    compile(const Slang::ComPtr<slang::IComponentType>& linked_programm,
            const uint32_t entrypoint_index);

    // This compiles all entrypoints. You can skip compose and directly link the module. This will
    // compile all entrypoints in the linked composite.
    static Slang::ComPtr<slang::IBlob>
    compile(const Slang::ComPtr<slang::IComponentType>& linked_programm);

    // Shortcut for compile() then ShaderModule::create.
    //
    // This compiles all entrypoints in the linked programm. You can skip compose and directly link
    // the module.
    static ShaderModuleHandle
    compile_to_shadermodule(const ContextHandle& context,
                            const Slang::ComPtr<slang::IComponentType>& linked_programm) {
        Slang::ComPtr<slang::IBlob> compiled = compile(linked_programm);

        return ShaderModule::create(context, compiled->getBufferPointer(),
                                    compiled->getBufferSize());
    }

    // Should only be used for very simple shader. Otherwise use the SlangComposition class.
    static EntryPointHandle
    compile_entry_point(const ContextHandle& context,
                        const Slang::ComPtr<slang::IComponentType>& linked_programm,
                        const int64_t& entry_point_index = 0) {
        Slang::ComPtr<slang::IBlob> compiled;
        Slang::ComPtr<slang::IBlob> diagnostics_blob;

        slang::EntryPointReflection* entry_point =
            linked_programm->getLayout()->getEntryPointByIndex(entry_point_index);
        if (entry_point == nullptr) {
            throw ShaderCompiler::compilation_failed{
                fmt::format("entry point with index {} does not exist", entry_point_index)};
        }

        SlangResult result = linked_programm->getEntryPointCode(
            entry_point_index, 0, // targetIndex, currently only one supported,
            compiled.writeRef(), diagnostics_blob.writeRef());

        if (SLANG_FAILED(result)) {
            throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
        }

        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang compiling. Diagnostics: {}",
                         diagnostics_as_string(diagnostics_blob));
        }

        return EntryPoint::create(
            entry_point->getNameOverride(), vk_stage_for_slang_stage(entry_point->getStage()),
            ShaderModule::create(context, compiled->getBufferPointer(), compiled->getBufferSize()));
    }

    // Should only be used for very simple shader. Otherwise use the SlangComposition class.
    static EntryPointHandle
    compile_entry_point(const ContextHandle& context,
                        const Slang::ComPtr<slang::IComponentType>& linked_program,
                        const std::string& entry_point_name = "main") {
        slang::ProgramLayout* layout = linked_program->getLayout();

        for (uint32_t i = 0; i < layout->getEntryPointCount(); i++) {
            if (entry_point_name == layout->getEntryPointByIndex(i)->getNameOverride()) {
                return compile_entry_point(context, linked_program, i);
            }
        }

        throw ShaderCompiler::compilation_failed{
            fmt::format("entry point with name {} does not exist", entry_point_name)};
    }

    // -----------------------------------------------------

    // Shortcut for load_module_from_path + compose_all_entrypoints + link + compile.
    // Should be only used for very simple cases otherwise use the SlangComposition class.
    EntryPointHandle load_module_from_path_and_compile_entry_point(
        const ContextHandle& context,
        const std::filesystem::path& path,
        const std::string& entry_point_name = "main",
        const std::optional<std::string>& relative_to = std::nullopt) {
        return load_module_from_path_and_compile_entry_point(context, path.stem().string(), path,
                                                             entry_point_name, relative_to);
    }

    // Shortcut for load_module_from_path + compose_all_entrypoints + link + compile.
    // Should be only used for very simple cases otherwise use the SlangComposition class.
    EntryPointHandle load_module_from_path_and_compile_entry_point(
        const ContextHandle& context,
        const std::string& name,
        const std::filesystem::path& path,
        const std::string& entry_point_name = "main",
        const std::optional<std::filesystem::path>& relative_to = std::nullopt) {
        Slang::ComPtr<slang::IModule> module = load_module_from_path(name, path, relative_to);
        return compile_entry_point(context, link(compose_all_entrypoints(module)),
                                   entry_point_name);
    }

    // Shortcut for load_module_from_source + compose_all_entrypoints + link + compile.
    // Should be only used for very simple cases otherwise use the SlangComposition class.
    EntryPointHandle load_module_from_source_and_compile_entry_point(
        const ContextHandle& context,
        const std::string& name,
        const std::string& source,
        const std::string& entry_point_name = "main",
        const std::optional<std::filesystem::path>& path = std::nullopt) {
        Slang::ComPtr<slang::IModule> module = load_module_from_source(name, source, path);

        return compile_entry_point(context, link(compose_all_entrypoints(module)),
                                   entry_point_name);
    }

    // Find the TypeLayoutReflection* for a named type in a composition or module.
    // Caller must keep the returned SlangProgramHandle alive to keep the pointer valid.
    struct TypeLayoutResult {
        slang::TypeLayoutReflection* type_layout;
        SlangProgramHandle program;
    };

    static TypeLayoutResult get_type_layout(const ShaderCompileContextHandle& compile_context,
                                            const SlangCompositionHandle& composition,
                                            const std::string& type_name);

    static TypeLayoutResult get_type_layout(const ShaderCompileContextHandle& compile_context,
                                            const std::filesystem::path& module_path,
                                            const std::string& type_name);

  public:
    // Thrown when a composition needs a fresh session.
    class stale_session : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

    static SlangSessionHandle create(const ShaderCompileContextHandle& shader_compile_context);

    // Runs the on-disk cache size-cap eviction (see cache_evict).
    ~SlangSession();

  private:
    static constexpr SlangDebugInfoLevel debug_info_level(const bool generate_debug_info) {
        return generate_debug_info ? SLANG_DEBUG_INFO_LEVEL_MAXIMAL : SLANG_DEBUG_INFO_LEVEL_NONE;
    }

    static std::string diagnostics_as_string(Slang::ComPtr<slang::IBlob>& diagnostics_blob) {
        if (diagnostics_blob == nullptr) {
            return {};
        }
        return (const char*)diagnostics_blob->getBufferPointer();
    }

    // --- on-disk shader cache (serialized IR modules + SPIR-V) ---

    // MERIAN_SHADER_CACHE != "0".
    static bool cache_enabled();
    // MERIAN_SHADER_CACHE_DIR, else <cwd>/.merian-cache.
    static const std::filesystem::path& cache_root();
    // Sanitized Slang build tag; the level under the root that isolates Slang versions.
    static const std::string& cache_tag();
    // <root>/<slang-build-tag>/<subdir>; the build-tag level isolates Slang versions.
    static std::filesystem::path cache_dir(std::string_view subdir);
    // File -> owning blob (nullptr if absent/unreadable); touches mtime on a hit.
    static Slang::ComPtr<slang::IBlob> cache_read(const std::filesystem::path& path);
    // Atomic write (temp + rename), best-effort.
    static void cache_write(const std::filesystem::path& path, const void* data, size_t size);
    // Drops other build tags, then an LRU size-cap sweep (MERIAN_SHADER_CACHE_MAX_MB,
    // default 512, 0 = unbounded).
    static void cache_evict();

    // One directory per compile configuration.
    static std::filesystem::path module_cache_dir(const ShaderCompileContext& context);
    // Presents cached binary modules as `.slang-module` files beside their sources.
    static Slang::ComPtr<ISlangFileSystemExt>
    create_file_system(const std::filesystem::path& module_cache_dir,
                       std::function<bool()> serve_binaries);
    // Writes the binaries of the file modules this session compiled.
    void store_modules();
    std::optional<std::filesystem::path>
    canonical_file_path(const std::filesystem::path& path) const;
    // The name of the file's module when imported by its canonical spelling, nullopt outside the
    // search paths.
    std::optional<std::string> canonical_module_name(const std::filesystem::path& file) const;
    // Slang names a module after the first import that reaches it, and mangled names contain the
    // module name: a binary cannot load against a dependency compiled under another name.
    bool modules_canonically_named();
    // Slang's own backend cache key (getEntryPointHash) as a filename, or nullopt to skip.
    static std::optional<std::filesystem::path>
    spirv_cache_path(const Slang::ComPtr<slang::IComponentType>& program);
    static std::optional<std::filesystem::path>
    spirv_cache_path(const Slang::ComPtr<slang::IComponentType>& program,
                     uint32_t entry_point_index);

    const Slang::ComPtr<slang::IModule>& ensure_module(SlangComposition::SlangModule& module);
    static void check_one_source_per_name(const SlangCompositionHandle& composition,
                                          std::unordered_map<std::string, uint64_t>& sources);
    Slang::ComPtr<slang::IComponentType> compose_tree(const SlangCompositionHandle& composition);
    bool is_imported(slang::IModule* module) const;
    bool is_replaced(slang::IModule* module) const;
    void check_no_replaced_dependency(slang::IModule* module) const;

  private:
    // The context owns the session it hands out, so the way back must not own it: a handle here
    // closes a reference cycle that keeps both alive past the last use, and with them the
    // destructor that caps the cache on disk.
    ShaderCompileContextHandle compile_context() const {
        return weak_compile_context.lock();
    }

    static constexpr std::size_t MAX_REPLACED_MODULES = 64;

    const std::weak_ptr<ShaderCompileContext> weak_compile_context;
    Slang::ComPtr<ISlangFileSystemExt> file_system;
    Slang::ComPtr<slang::ISession> session;

    // -> entry_point, renamed
    std::map<SlangComposition::EntryPoint,
             std::pair<Slang::ComPtr<slang::IEntryPoint>, Slang::ComPtr<slang::IComponentType>>>
        entry_point_cache;
    // identity (the path Slang reports as dependency) -> module
    std::unordered_map<std::string, slang::IModule*> source_strings_by_identity;
    std::unordered_set<slang::IModule*> checked_module_names;
    bool module_names_canonical = true;
    struct BoundModule {
        uint64_t source_hash;
        Slang::ComPtr<slang::IModule> module;
    };
    std::unordered_map<std::string, BoundModule> bound_modules;
    std::map<std::pair<std::string, uint64_t>, Slang::ComPtr<slang::IModule>> module_versions;
    std::map<SlangComposition::TypeConformance, Slang::ComPtr<slang::IComponentType>>
        type_conformance_cache;
    // -> composition version, composed
    std::unordered_map<SlangCompositionHandle,
                       std::pair<uint64_t, Slang::ComPtr<slang::IComponentType>>>
        composition_cache;
};

} // namespace merian
