#pragma once

#include "merian/vk/shader/glsl_shader_compiler.hpp"

namespace merian {

// Uses glslangValidator executable to compile shaders.
class SystemGlslangValidatorCompiler : public GLSLShaderCompiler {
  public:
    // Include paths for the merian-nodes library are automatically added
    SystemGlslangValidatorCompiler();

    ~SystemGlslangValidatorCompiler();

    std::vector<uint32_t> compile_glsl(
        const std::string& source,
        const std::string& source_name,
        const vk::ShaderStageFlagBits shader_kind,
        const CompilationSessionDescription& compilation_session_description) const override;

    bool available() const override;

  private:
    const std::string compiler_executable;
};

} // namespace merian
