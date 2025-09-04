#pragma once

#include "merian/vk/shader/entry_point.hpp"
#include "merian/vk/shader/shader_compile_context.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_module.hpp"
#include "merian/vk/shader/slang_composition.hpp"
#include "merian/vk/shader/slang_global_session.hpp"

#include "slang-com-ptr.h"
#include "slang.h"
#include <ranges>

namespace merian {

class SlangSession;
using SlangSessionHandle = std::shared_ptr<SlangSession>;

// A wrapper around a slang session.
class SlangSession {
  protected:
    SlangSession(const ShaderCompileContextHandle& shader_compile_context)
        : shader_compile_context(shader_compile_context) {
        const auto global_session = get_global_slang_session();

        slang::SessionDesc slang_session_desc = {};

        slang::TargetDesc target_desc = {};
        switch (shader_compile_context->get_target()) {
        case CompilationTarget::SPIRV_1_0:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_0");
            break;
        case CompilationTarget::SPIRV_1_1:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_1");
            break;
        case CompilationTarget::SPIRV_1_2:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_2");
            break;
        case CompilationTarget::SPIRV_1_3:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_3");
            break;
        case CompilationTarget::SPIRV_1_4:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_4");
            break;
        case CompilationTarget::SPIRV_1_5:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_5");
            break;
        case CompilationTarget::SPIRV_1_6:
            target_desc.format = SLANG_SPIRV;
            target_desc.profile = global_session->findProfile("spirv_1_6");
            break;
        default:
            throw std::runtime_error{"Target not supported"};
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

        std::vector<const char*> search_paths;
        search_paths.reserve(shader_compile_context->get_preprocessor_macros().size());
        for (const auto& search_path : shader_compile_context->get_search_path_file_loader()) {
            search_paths.emplace_back(search_path.c_str());
        }
        slang_session_desc.searchPaths = search_paths.data();
        slang_session_desc.searchPathCount = (SlangInt)search_paths.size();

        std::array<slang::CompilerOptionEntry, 2> options = {
            {
                {
                    slang::CompilerOptionName::EmitSpirvDirectly,
                    {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
                },
                {
                    slang::CompilerOptionName::Optimization,
                    {slang::CompilerOptionValueKind::Int,
                     static_cast<int32_t>(shader_compile_context->get_optimization_level()), 0,
                     nullptr, nullptr},
                },
            },
        };
        slang_session_desc.compilerOptionEntries = options.data();
        slang_session_desc.compilerOptionEntryCount = options.size();

        global_session->createSession(slang_session_desc, session.writeRef());
    }

  public:
    const ShaderCompileContextHandle& get_compile_context() {
        return shader_compile_context;
    }

    // The path can be used as path-based import statement the
    // module. The name is the stem (final part without its suffix) of this path. If the source path
    // should not be the same as the path for path-based includes use "source_path".
    //
    // Note: The returned module is only valid as long as this session is valid
    Slang::ComPtr<slang::IModule>
    load_module_from_path(const std::filesystem::path& path,
                          const std::optional<std::filesystem::path>& source_path = std::nullopt) {
        return load_module_from_path(path.stem(), path, source_path);
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
        std::optional<std::string> source;
        if (source_path) {
            source = shader_compile_context->get_search_path_file_loader().find_and_load_file(
                source_path.value());
        } else {
            source = shader_compile_context->get_search_path_file_loader().find_and_load_file(path);
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
                            const std::optional<std::filesystem::path>& path) {
        Slang::ComPtr<slang::IBlob> diagnostics_blob;
        Slang::ComPtr<slang::IModule> module;
        module = session->loadModuleFromSourceString(name.c_str(), path ? path->c_str() : nullptr,
                                                     source.c_str(), diagnostics_blob.writeRef());

        if (module == nullptr) {
            throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
        }

        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang compiling module {} ({}). Diagnostics: {}", name,
                         path.has_value() ? path->string() : "no path",
                         diagnostics_as_string(diagnostics_blob));
        }

        return module;
    }

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

    Slang::ComPtr<slang::IComponentType> compose(const SlangCompositionHandle& composition) {
        std::vector<slang::IComponentType*> components;
        components.reserve(
            std::max(composition->modules.size() + composition->compositions.size(),
                     1 + composition->type_conformances.size() + composition->entry_points.size()));

        std::set<SlangComposition::EntryPoint> additional_entry_points;
        for (auto& [_, module] : composition->modules) {
            auto it = slang_module_cache.find(module.get_name());
            if (it == slang_module_cache.end()) {
                it = slang_module_cache
                         .emplace(module.get_name(),
                                  load_module_from_source(
                                      module.get_name(),
                                      module.get_source(
                                          get_compile_context()->get_search_path_file_loader()),
                                      module.get_import_path()))
                         .first;
            }

            components.emplace_back(it->second);

            if (module.get_with_entry_points()) {
                for (uint32_t entry_point_index = 0;
                     entry_point_index <
                     merian::SlangSession::get_defined_entry_point_count(it->second);
                     entry_point_index++) {
                    Slang::ComPtr<slang::IEntryPoint> entry_point =
                        merian::SlangSession::get_defined_entry_point(it->second,
                                                                      entry_point_index);
                    const char* name = entry_point->getFunctionReflection()->getName();
                    auto rename_it = module.get_entry_point_map().find(name);
                    if (rename_it == module.get_entry_point_map().end()) {
                        additional_entry_points.insert(
                            SlangComposition::EntryPoint(name, module.get_name()));
                    } else {
                        additional_entry_points.insert(SlangComposition::EntryPoint(
                            name, module.get_name(), rename_it->second));
                    }
                }
            }
        }

        for (const auto& composition : composition->compositions) {
            auto it = composition_cache.find(composition);
            if (it == composition_cache.end()) {
                it = composition_cache.emplace(composition, compose(composition)).first;
            }
            components.emplace_back(it->second);
        }

        Slang::ComPtr<slang::IComponentType> composed_modules = compose(components);
        components.clear();
        components.emplace_back(composed_modules);

        for (auto& [type_conformance, c_id] : composition->type_conformances) {
            auto it = type_conformance_cache.find(type_conformance);
            if (it == type_conformance_cache.end()) {
                int64_t id = c_id;
                it = type_conformance_cache
                         .emplace(type_conformance,
                                  create_type_conformance(
                                      composed_modules, type_conformance.get_type_name(),
                                      type_conformance.get_interface_name(), id))
                         .first;
            }
            components.emplace_back(it->second);
        }

        for (const auto& ep :
             std::views::join(std::array{std::views::all(composition->entry_points),
                                         std::views::all(additional_entry_points)})) {
            auto it = entry_point_cache.find(ep);
            if (it == entry_point_cache.end()) {
                auto& module = slang_module_cache.at(ep.get_module());

                it = entry_point_cache
                         .emplace(ep, std::make_pair(merian::SlangSession::find_entry_point_or_fail(
                                                         module, ep.get_defined_name()),
                                                     nullptr))
                         .first;

                it->second.first->renameEntryPoint(ep.get_export_name().c_str(),
                                                   it->second.second.writeRef());
            }
            components.push_back(it->second.second);
        }

        return compose(components);
    }

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
            const uint32_t entrypoint_index) {
        Slang::ComPtr<slang::IBlob> compiled;
        Slang::ComPtr<slang::IBlob> diagnostics_blob;

        SlangResult result =
            linked_programm->getEntryPointCode(entrypoint_index,
                                               0, // targetIndex, currently only one supported
                                               compiled.writeRef(), diagnostics_blob.writeRef());

        if (SLANG_FAILED(result)) {
            throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
        }

        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang compiling. Diagnostics: {}",
                         diagnostics_as_string(diagnostics_blob));
        }

        return compiled;
    }

    // This compiles all entrypoints. You can skip compose and directly link the module. This will
    // compile all entrypoints in the linked composite.
    static Slang::ComPtr<slang::IBlob>
    compile(const Slang::ComPtr<slang::IComponentType>& linked_programm) {
        Slang::ComPtr<slang::IBlob> compiled;
        Slang::ComPtr<slang::IBlob> diagnostics_blob;

        SlangResult result =
            linked_programm->getTargetCode(0, // targetIndex, currently only one supported,
                                           compiled.writeRef(), diagnostics_blob.writeRef());

        if (SLANG_FAILED(result)) {
            throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
        }

        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang compiling. Diagnostics: {}",
                         diagnostics_as_string(diagnostics_blob));
        }

        return compiled;
    }

    // This compiles all entrypoints in the linked programm. You can skip compose and directly link
    // the module.
    //
    // Should only be used for very simple shader. Otherwise use the SlangComposition class.
    static ShaderModuleHandle
    compile_to_shadermodule(const ContextHandle& context,
                            const Slang::ComPtr<slang::IComponentType>& linked_programm) {
        Slang::ComPtr<slang::IBlob> compiled;
        Slang::ComPtr<slang::IBlob> diagnostics_blob;

        SlangResult result =
            linked_programm->getTargetCode(0, // targetIndex, currently only one supported,
                                           compiled.writeRef(), diagnostics_blob.writeRef());

        if (SLANG_FAILED(result)) {
            throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
        }

        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang compiling. Diagnostics: {}",
                         diagnostics_as_string(diagnostics_blob));
        }

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
        return load_module_from_path_and_compile_entry_point(context, path.stem(), path,
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

  public:
    static SlangSessionHandle create(const ShaderCompileContextHandle& shader_compile_context);

    // returns a cached session for the context or creates one if none is avaiable.
    static SlangSessionHandle
    get_or_create(const ShaderCompileContextHandle& shader_compile_context);

  private:
    static std::string diagnostics_as_string(Slang::ComPtr<slang::IBlob>& diagnostics_blob) {
        if (diagnostics_blob == nullptr) {
            return {};
        }
        return (const char*)diagnostics_blob->getBufferPointer();
    }

  private:
    const ShaderCompileContextHandle shader_compile_context;
    Slang::ComPtr<slang::ISession> session;

    // -> entry_point, renamed
    std::map<SlangComposition::EntryPoint,
             std::pair<Slang::ComPtr<slang::IEntryPoint>, Slang::ComPtr<slang::IComponentType>>>
        entry_point_cache;
    std::map<std::string, Slang::ComPtr<slang::IModule>> slang_module_cache;
    std::map<SlangComposition::TypeConformance, Slang::ComPtr<slang::IComponentType>>
        type_conformance_cache;
    std::map<SlangCompositionHandle, Slang::ComPtr<slang::IComponentType>> composition_cache;
};

} // namespace merian
