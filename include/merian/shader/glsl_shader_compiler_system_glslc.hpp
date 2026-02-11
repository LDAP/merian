#pragma once

#include "merian/shader/glsl_shader_compiler.hpp"

namespace merian {

// Uses shaderc executable to compile shaders.
class SystemGlslcCompiler : public GLSLShaderCompiler {
  public:
    // Include paths for the merian-nodes library are automatically added
    SystemGlslcCompiler();

    ~SystemGlslcCompiler();

    BlobHandle
    compile_glsl(const std::string& source,
                 const std::string& source_name,
                 const vk::ShaderStageFlagBits shader_kind,
                 const ShaderCompileContextHandle& shader_compile_context) const override;

    bool available() const override;

  private:
    const std::string compiler_executable;
};

} // namespace merian
