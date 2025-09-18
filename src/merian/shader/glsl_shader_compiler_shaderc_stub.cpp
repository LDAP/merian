#include "merian/shader/glsl_shader_compiler_shaderc.hpp"

namespace merian {

ShadercCompiler::ShadercCompiler() : GLSLShaderCompiler() {}

ShadercCompiler::~ShadercCompiler() {}

std::vector<uint32_t> ShadercCompiler::compile_glsl(
    [[maybe_unused]] const std::string& source,
    [[maybe_unused]] const std::string& source_name,
    [[maybe_unused]] const vk::ShaderStageFlagBits shader_kind,
    [[maybe_unused]] const ShaderCompileContextHandle& shader_compile_context) const {
    throw merian::ShaderCompiler::compilation_failed{
        "shaderc is not available (was not found or enabled at compile time)"};
}

bool ShadercCompiler::available() const {
    return false;
}

} // namespace merian
