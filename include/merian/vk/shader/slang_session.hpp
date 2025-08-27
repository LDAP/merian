#pragma once

#include "merian/vk/shader/compilation_session_description.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/slang_global_session.hpp"

#include "slang-com-ptr.h"
#include "slang.h"

namespace merian {

class SlangSession {
  public:
    SlangSession(const CompilationSessionDescription& compilation_session_description) {
        const auto global_session = get_global_slang_session();

        slang::SessionDesc slang_session_desc = {};

        slang::TargetDesc target_desc = {};
        switch (compilation_session_description.get_target()) {
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
        preprocessor_macros.reserve(
            compilation_session_description.get_preprocessor_macros().size());

        for (const auto& macro : compilation_session_description.get_preprocessor_macros()) {
            preprocessor_macros.emplace_back(macro.first.c_str(), macro.second.c_str());
        }
        slang_session_desc.preprocessorMacros = preprocessor_macros.data();
        slang_session_desc.preprocessorMacroCount = (SlangInt)preprocessor_macros.size();

        std::vector<const char*> search_paths;
        search_paths.reserve(compilation_session_description.get_preprocessor_macros().size());
        for (const auto& search_path : compilation_session_description.get_include_paths()) {
            search_paths.emplace_back(search_path.c_str());
        }
        slang_session_desc.searchPaths = search_paths.data();
        slang_session_desc.searchPathCount = (SlangInt)search_paths.size();
        file_loader.add_search_path(compilation_session_description.get_include_paths());

        std::array<slang::CompilerOptionEntry, 1> options = {
            {
                {slang::CompilerOptionName::EmitSpirvDirectly,
                 {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
            },
        };
        slang_session_desc.compilerOptionEntries = options.data();
        slang_session_desc.compilerOptionEntryCount = options.size();

        get_global_slang_session()->createSession(slang_session_desc, session.writeRef());
    }

    // Loads a module from this path. The path can be used as path-based import statement the
    // module. The name is the stem (final part without its suffix) of this path.
    //
    // Note: The returned module is only valid as long as this session is valid
    Slang::ComPtr<slang::IModule>
    load_module_from_path(const std::filesystem::path& path,
                          const std::optional<std::string>& relative_to = std::nullopt) {
        return load_module_from_path(path.stem(), path, relative_to);
    }

    // Loads a module from this path. The path can be used as path-based import statement the
    // module.
    //
    // Note: The returned module is only valid as long as this session is valid
    Slang::ComPtr<slang::IModule>
    load_module_from_path(const std::string& name,
                          const std::filesystem::path& path,
                          const std::optional<std::filesystem::path>& relative_to = std::nullopt) {
        std::optional<std::string> source;
        if (relative_to) {
            source = file_loader.find_and_load_file(path, relative_to.value());
        } else {
            source = file_loader.find_and_load_file(path);
        }

        if (!source) {
            throw ShaderCompiler::compilation_failed(
                fmt::format("Compiling module {} from {} failed: Not found", name, path));
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
            SPDLOG_DEBUG("Slang compiling module {} ({}). Diagnostics: {}", name, path,
                         diagnostics_as_string(diagnostics_blob));
        }

        return module;
    }

    Slang::ComPtr<slang::IEntryPoint> find_entry_point(Slang::ComPtr<slang::IModule>& module,
                                                       const std::string& name) {
        Slang::ComPtr<slang::IEntryPoint> entry_point;
        module->findEntryPointByName(name.c_str(), entry_point.writeRef());
        return entry_point;
    }

    // throws compilaiton failed if not found
    Slang::ComPtr<slang::IEntryPoint>
    find_entry_point_or_fail(Slang::ComPtr<slang::IModule>& module, const std::string& name) {
        Slang::ComPtr<slang::IEntryPoint> entry_point = find_entry_point(module, name);
        if (entry_point == nullptr) {
            throw ShaderCompiler::compilation_failed(fmt::format(
                "entrypoint {} could not be found in module {}", name, module->getName()));
        }
        return entry_point;
    }

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

    Slang::ComPtr<slang::IBlob> compile(const Slang::ComPtr<slang::IComponentType>& linked_programm,
                                        uint32_t entrypoint_index) {
        Slang::ComPtr<slang::IBlob> compiled;
        Slang::ComPtr<slang::IBlob> diagnostics_blob;

        SlangResult result =
            linked_programm->getEntryPointCode(0, // entryPointIndex
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
    // compile all entrypoints.
    Slang::ComPtr<slang::IBlob>
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

  private:
    static std::string diagnostics_as_string(Slang::ComPtr<slang::IBlob>& diagnostics_blob) {
        if (diagnostics_blob == nullptr) {
            return {};
        }
        return (const char*)diagnostics_blob->getBufferPointer();
    }

  private:
    Slang::ComPtr<slang::ISession> session;
    FileLoader file_loader;
};

} // namespace merian
