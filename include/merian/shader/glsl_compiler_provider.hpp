#pragma once

#include "merian/shader/glsl_shader_compiler.hpp"

namespace merian {

class GLSLCompilerProvider {
  public:
    virtual ~GLSLCompilerProvider() = default;
    virtual GLSLShaderCompilerHandle get_glsl_compiler() const = 0;
};

} // namespace merian
