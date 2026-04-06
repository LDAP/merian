#include "merian/shader/slang_session.hpp"
#include "merian/shader/slang_program.hpp"

namespace merian {

static std::map<ShaderCompileContextHandle, std::weak_ptr<SlangSession>> cached_session_for_context;
static std::map<SlangSession*, ShaderCompileContextHandle> cached_context_for_session;

SlangSessionHandle SlangSession::create(const ShaderCompileContextHandle& shader_compile_context) {
    SlangSession* session = new SlangSession(shader_compile_context);
    cached_context_for_session[session] = shader_compile_context;
    SlangSessionHandle shared_session = SlangSessionHandle(session, [](SlangSession* p) {
        auto ctx_it = cached_context_for_session.find(p);
        if (ctx_it != cached_context_for_session.end()) {
            // Only erase the cache entry if it still points to this session
            auto cache_it = cached_session_for_context.find(ctx_it->second);
            if (cache_it != cached_session_for_context.end() && cache_it->second.expired()) {
                cached_session_for_context.erase(cache_it);
            }
            cached_context_for_session.erase(ctx_it);
        }
        SPDLOG_DEBUG("erase slang session from cache");
        delete p;
    });
    cached_session_for_context[shader_compile_context] = shared_session;
    return shared_session;
}

// returns a cached session for the context or creates one if none is avaiable.
SlangSessionHandle
SlangSession::get_or_create(const ShaderCompileContextHandle& shader_compile_context,
                            const bool force_new) {
    if (!force_new) {
        auto it = cached_session_for_context.find(shader_compile_context);
        if (it != cached_session_for_context.end() && !it->second.expired()) {
            SPDLOG_DEBUG("reuse slang session from cache");
            return it->second.lock();
        }
    }
    return create(shader_compile_context);
}
static slang::TypeLayoutReflection*
find_type_layout(slang::ProgramLayout* layout, const std::string& type_name) {
    slang::TypeReflection* type = layout->findTypeByName(type_name.c_str());
    if (type == nullptr) {
        throw ShaderCompiler::compilation_failed(
            fmt::format("type '{}' not found", type_name));
    }

    slang::TypeLayoutReflection* type_layout =
        layout->getTypeLayout(type, slang::LayoutRules::Default);
    if (type_layout == nullptr) {
        throw ShaderCompiler::compilation_failed(
            fmt::format("failed to get type layout for '{}'", type_name));
    }

    return type_layout;
}

SlangSession::TypeLayoutResult
SlangSession::get_type_layout(const ShaderCompileContextHandle& compile_context,
                              const SlangCompositionHandle& composition,
                              const std::string& type_name) {
    auto program = SlangProgram::create(compile_context, composition);
    auto* type_layout = find_type_layout(program->get_program_reflection(), type_name);
    return {type_layout, program};
}

SlangSession::TypeLayoutResult
SlangSession::get_type_layout(const ShaderCompileContextHandle& compile_context,
                              const std::filesystem::path& module_path,
                              const std::string& type_name) {
    auto program = SlangProgram::create(compile_context, module_path, false);
    auto* type_layout = find_type_layout(program->get_program_reflection(), type_name);
    return {type_layout, program};
}

} // namespace merian
