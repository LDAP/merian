#pragma once

#include "merian/shader/glsl_shader_compiler.hpp"

#ifdef MERIAN_SHADERC_AVAILABLE
#include <shaderc/shaderc.hpp>
#endif

namespace merian {

class ShadercCompiler : public GLSLShaderCompiler {
  public:
    ShadercCompiler();

    ~ShadercCompiler();

    BlobHandle
    compile_glsl(const std::string& source,
                 const std::string& source_name,
                 const vk::ShaderStageFlagBits shader_kind,
                 const ShaderCompileContextHandle& shader_compile_context) const override;

    bool available() const override;

  private:
#ifdef MERIAN_SHADERC_AVAILABLE
    shaderc::Compiler shader_compiler{};
#endif
};

} // namespace merian
