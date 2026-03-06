#pragma once

#include "merian/shader/glsl_shader_compiler.hpp"

namespace merian {

// Uses linked glslang to compile shaders.
class GlslangCompiler : public GLSLShaderCompiler {
  public:
    GlslangCompiler();

    ~GlslangCompiler();

    BlobHandle
    compile_glsl(const std::string& source,
                 const std::string& source_name,
                 const vk::ShaderStageFlagBits shader_kind,
                 const ShaderCompileContextHandle& shader_compile_context) const override;

    bool available() const override;
};

} // namespace merian
