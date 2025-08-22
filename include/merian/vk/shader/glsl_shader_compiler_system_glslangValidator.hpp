#pragma once

#include "merian/vk/shader/glsl_shader_compiler.hpp"
#include <map>

namespace merian {

// Uses glslangValidator executable to compile shaders.
class SystemGlslangValidatorCompiler : public GLSLShaderCompiler {
  public:
    // Include paths for the merian-nodes library are automatically added
    SystemGlslangValidatorCompiler(
        const ContextHandle& context,
        const std::vector<std::string>& include_paths = {},
        const std::map<std::string, std::string>& macro_definitions = {});

    ~SystemGlslangValidatorCompiler();

    std::vector<uint32_t> compile_glsl(
        const std::string& source,
        const std::string& source_name,
        const vk::ShaderStageFlagBits shader_kind,
        const std::vector<std::string>& additional_include_paths = {},
        const std::map<std::string, std::string>& additional_macro_definitions = {}) const override;

    bool available() const override;

  private:
    const std::string compiler_executable;
    std::string target_env_arg;
};

} // namespace merian
