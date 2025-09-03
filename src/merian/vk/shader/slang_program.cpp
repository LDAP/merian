#include "merian/vk/shader/slang_program.hpp"
#include "merian/vk/shader/slang_entry_point.hpp"

namespace merian {

SlangProgram::SlangProgram(const SlangCompositionHandle& composition) : composition(composition) {
    program = merian::SlangSession::link(composition->get_composite());
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
            get_program_reflection()->getEntryPointByIndex(entry_point_index)->getName()) {
            return entry_point_index;
        }
    }

    throw std::invalid_argument{
        fmt::format("no entry point with name {} in program", entry_point_name)};
}

SlangEntryPointHandle SlangProgram::get_entry_point_by_index(const uint64_t entry_point_index) {
    return SlangEntryPoint::create(shared_from_this(), entry_point_index);
}

SlangEntryPointHandle SlangProgram::get_entry_point_by_name(const std::string& entry_point_name) {
    return get_entry_point_by_index(get_entry_point_index(entry_point_name));
}

const SlangCompositionHandle& SlangProgram::get_composition() {
    return composition;
}

SlangProgramHandle SlangProgram::create(const SlangCompositionHandle& composition) {
    return SlangProgramHandle(new SlangProgram(composition));
}

} // namespace merian
