#include "merian/vk/shader/slang_program.hpp"

namespace merian {

SlangProgram::SlangProgram(const ShaderCompileContextHandle& compile_context,
                           const SlangCompositionHandle& composition)
    : compile_context(compile_context), composition(composition) {

    session = SlangSession::get_or_create(compile_context);
    program = merian::SlangSession::link(session->compose(composition));
}

ShaderModuleHandle SlangProgram::get_shader_module(const ContextHandle& context) {
    if (!shader_module) {
        shader_module = merian::SlangSession::compile_to_shadermodule(context, program);
    }

    return shader_module;
}

slang::ProgramLayout* SlangProgram::get_program_reflection() const {
    return program->getLayout();
}

const Slang::ComPtr<slang::IComponentType>& SlangProgram::get_program() const {
    return program;
}

uint64_t SlangProgram::get_entry_point_index(const std::string& entry_point_name) const {
    for (uint64_t entry_point_index = 0;
         entry_point_index < get_program_reflection()->getEntryPointCount(); entry_point_index++) {
        if (entry_point_name ==
            get_program_reflection()->getEntryPointByIndex(entry_point_index)->getNameOverride()) {
            return entry_point_index;
        }
    }

    throw std::invalid_argument{
        fmt::format("no entry point with name {} in program", entry_point_name)};
}

const SlangCompositionHandle& SlangProgram::get_composition() {
    return composition;
}

SlangProgramHandle SlangProgram::create(const ShaderCompileContextHandle& compile_context,
                                        const SlangCompositionHandle& composition) {
    return SlangProgramHandle(new SlangProgram(compile_context, composition));
}

SlangProgramHandle SlangProgram::create(const ShaderCompileContextHandle& compile_context,
                                        const std::filesystem::path& path,
                                        const bool with_entry_points) {
    const SlangCompositionHandle comp = SlangComposition::create();
    comp->add_module_from_path(path, with_entry_points);
    return create(compile_context, comp);
}

} // namespace merian
