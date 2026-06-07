#include "merian/shader/slang_session.hpp"
#include "merian/shader/slang_program.hpp"

namespace merian {

SlangSessionHandle SlangSession::create(const ShaderCompileContextHandle& shader_compile_context) {
    SPDLOG_DEBUG("create slang session");
    return SlangSessionHandle(new SlangSession(shader_compile_context));
}

SlangSessionHandle ShaderCompileContext::current_session() {
    const uint64_t epoch = slang_source_epoch();
    if (!hot_session || hot_session_epoch != epoch) {
        hot_session = SlangSession::create(shared_from_this());
        hot_session_epoch = epoch;
    }
    return hot_session;
}

static slang::TypeLayoutReflection* find_type_layout(slang::ProgramLayout* layout,
                                                     const std::string& type_name) {
    slang::TypeReflection* type = layout->findTypeByName(type_name.c_str());
    if (type == nullptr) {
        throw ShaderCompiler::compilation_failed(fmt::format("type '{}' not found", type_name));
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
    const SlangProgramHandle program = SlangProgram::create(compile_context, composition).get();
    auto* type_layout = find_type_layout(program->get_program_reflection(), type_name);
    return {type_layout, program};
}

SlangSession::TypeLayoutResult
SlangSession::get_type_layout(const ShaderCompileContextHandle& compile_context,
                              const std::filesystem::path& module_path,
                              const std::string& type_name) {
    const SlangProgramHandle program =
        SlangProgram::create(compile_context, module_path, false).get();
    auto* type_layout = find_type_layout(program->get_program_reflection(), type_name);
    return {type_layout, program};
}

} // namespace merian
