#pragma once

#include "shader_compiler.hpp"

#include <memory>
#include <slang-com-ptr.h>
#include <slang.h>

namespace merian {

class SlangSharedLibrary;
typedef std::shared_ptr<SlangSharedLibrary> SlangSharedLibraryHandle;

class SlangSharedLibrary {
public:
    SlangSharedLibrary(const std::string& path) {
        loadSharedLibrary(path);
    }

    void loadSharedLibrary(const std::string& path) {
        Slang::ComPtr<slang::IGlobalSession> global_session;
        createGlobalSession(global_session.writeRef());

        Slang::ComPtr<slang::ICompileRequest> request;
        global_session->createCompileRequest(request.writeRef());

        const int targetIndex = request->addCodeGenTarget(SLANG_SHADER_HOST_CALLABLE);
        // Set the target flag to indicate that we want to compile all into a library.
        request->setTargetFlags(targetIndex, SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM);

        request->setOptimizationLevel(SLANG_OPTIMIZATION_LEVEL_NONE);
        request->setDebugInfoLevel(SLANG_DEBUG_INFO_LEVEL_STANDARD);

        const int translationUnitIndex =
            request->addTranslationUnit(SLANG_SOURCE_LANGUAGE_SLANG, nullptr);

        // Set the source file for the translation unit
        request->addTranslationUnitSourceFile(translationUnitIndex, path.c_str());

        const SlangResult compileRes = request->compile();

        if (auto diagnostics = request->getDiagnosticOutput())
        {
            printf("%s", diagnostics);
        }

        {
            SlangResult result = request->getTargetHostCallable(0, shared_lib.writeRef());
            if (SLANG_FAILED(result)) {
                throw merian::ShaderCompiler::compilation_failed("Failed to retrieve shared library");
            }
        }
    }

    template<typename FuncT>
    FuncT getFunctionByName(const std::string& name) const {
        SlangFuncPtr raw = shared_lib->findFuncByName(name.c_str());

        if (!raw)
        {
            throw merian::ShaderCompiler::compilation_failed("Failed to find function named " + name);
        }
        return reinterpret_cast<FuncT>(raw);
    }

public:
    static SlangSharedLibraryHandle create(const std::string& path) {
        return std::make_shared<SlangSharedLibrary>(path);
    }

private:
    Slang::ComPtr<ISlangSharedLibrary> shared_lib;

};

} // namespace merian