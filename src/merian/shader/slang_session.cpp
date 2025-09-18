#include "merian/shader/slang_session.hpp"

namespace merian {

static std::map<ShaderCompileContextHandle, std::weak_ptr<SlangSession>> cached_session_for_context;
static std::map<SlangSession*, ShaderCompileContextHandle> cached_context_for_session;

SlangSessionHandle SlangSession::create(const ShaderCompileContextHandle& shader_compile_context) {
    SlangSession* session = new SlangSession(shader_compile_context);
    cached_context_for_session[session] = shader_compile_context;
    SlangSessionHandle shared_session = SlangSessionHandle(session, [](SlangSession* p) {
        cached_session_for_context.erase(cached_context_for_session.at(p));
        cached_context_for_session.erase(p);
        SPDLOG_DEBUG("erase slang session from cache");

        delete p;
    });
    cached_session_for_context[shader_compile_context] = shared_session;
    return shared_session;
}

// returns a cached session for the context or creates one if none is avaiable.
SlangSessionHandle
SlangSession::get_or_create(const ShaderCompileContextHandle& shader_compile_context) {
    auto it = cached_session_for_context.find(shader_compile_context);
    if (it != cached_session_for_context.end() && !it->second.expired()) {
        SPDLOG_DEBUG("reuse slang session from cache");

        return it->second.lock();
    }
    return create(shader_compile_context);
}
} // namespace merian
