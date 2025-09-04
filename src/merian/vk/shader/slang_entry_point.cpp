#include "merian/vk/shader/slang_entry_point.hpp"
#include "merian/vk/shader/slang_global_session.hpp"

namespace merian {

SlangProgramEntryPoint::SlangProgramEntryPoint(const SlangProgramHandle& program,
                                               const uint64_t entry_point_index)
    : program(program), entry_point_index(entry_point_index) {
    assert(entry_point_index < program->get_program_reflection()->getEntryPointCount());
}

const char* SlangProgramEntryPoint::get_name() const {
    return get_entry_point_reflection()->getNameOverride();
}

vk::ShaderStageFlagBits SlangProgramEntryPoint::get_stage() const {
    return vk_stage_for_slang_stage(get_entry_point_reflection()->getStage());
}

ShaderModuleHandle
SlangProgramEntryPoint::vulkan_shader_module(const ContextHandle& context) const {
    return program->get_shader_module(context);
}

slang::EntryPointReflection* SlangProgramEntryPoint::get_entry_point_reflection() const {
    return program->get_program_reflection()->getEntryPointByIndex(entry_point_index);
}

const SlangProgramHandle& SlangProgramEntryPoint::get_program() const {
    return program;
}

SlangProgramEntryPointHandle SlangProgramEntryPoint::create(const SlangProgramHandle& program,
                                                            const uint64_t entry_point_index) {
    return SlangProgramEntryPointHandle(new SlangProgramEntryPoint(program, entry_point_index));
}

SlangProgramEntryPointHandle SlangProgramEntryPoint::create(const SlangProgramHandle& program,
                                                            const std::string& entry_point_name) {
    return create(program, program->get_entry_point_index(entry_point_name));
}

SlangProgramEntryPointHandle
SlangProgramEntryPoint::create(const ShaderCompileContextHandle& compile_context,
                               const std::filesystem::path& module_path,
                               const std::string& entry_point_name) {
    SlangProgramHandle program = SlangProgram::create(compile_context, module_path, true);
    return create(program, entry_point_name);
}

} // namespace merian
