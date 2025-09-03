#include "merian/vk/shader/slang_entry_point.hpp"
#include "merian/vk/shader/slang_global_session.hpp"

namespace merian {

SlangEntryPoint::SlangEntryPoint(const SlangProgramHandle& program,
                                 const uint64_t entry_point_index)
    : program(program), entry_point_index(entry_point_index) {
    assert(entry_point_index < program->get_program_reflection()->getEntryPointCount());
}

const char* SlangEntryPoint::get_name() const {
    return get_entry_point_reflection()->getName();
}

vk::ShaderStageFlagBits SlangEntryPoint::get_stage() const {
    return vk_stage_for_slang_stage(get_entry_point_reflection()->getStage());
}

ShaderModuleHandle SlangEntryPoint::vulkan_shader_module(const ContextHandle& context) const {
    return program->get_shader_module(context);
}

slang::EntryPointReflection* SlangEntryPoint::get_entry_point_reflection() const {
    return program->get_program_reflection()->getEntryPointByIndex(entry_point_index);
}

const SlangProgramHandle& SlangEntryPoint::get_program() const {
    return program;
}

SlangEntryPointHandle SlangEntryPoint::create(const SlangProgramHandle& program,
                                              const uint64_t entry_point_index) {
    return SlangEntryPointHandle(new SlangEntryPoint(program, entry_point_index));
}

SlangEntryPointHandle SlangEntryPoint::create(const SlangProgramHandle& program,
                                              const std::string& entry_point_name) {
    return create(program, program->get_entry_point_index(entry_point_name));
}

} // namespace merian
