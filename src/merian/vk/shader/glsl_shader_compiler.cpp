#include "merian/vk/shader/shader_compiler.hpp"

#include "merian/vk/shader/glsl_shader_compiler_shaderc.hpp"
#include "merian/vk/shader/glsl_shader_compiler_system_glslangValidator.hpp"
#include "merian/vk/shader/glsl_shader_compiler_system_glslc.hpp"

namespace merian {

class DummyGLSLShaderCompiler : public GLSLShaderCompiler {
  public:
    DummyGLSLShaderCompiler() : GLSLShaderCompiler() {}

    ~DummyGLSLShaderCompiler() {}

    std::vector<uint32_t> compile_glsl(
        [[maybe_unused]] const std::string& source,
        [[maybe_unused]] const std::string& source_name,
        [[maybe_unused]] const vk::ShaderStageFlagBits shader_kind,
        [[maybe_unused]] const ShaderCompileContextHandle& shader_compile_context) const override {
        throw compilation_failed{"compiler not available"};
    }

    bool available() const override {
        return false;
    }
};

static WeakGLSLShaderCompilerHandle cached_compiler;

GLSLShaderCompilerHandle GLSLShaderCompiler::get() {
    if (!cached_compiler.expired()) {
        return cached_compiler.lock();
    }

    GLSLShaderCompilerHandle shaderc = std::make_shared<ShadercCompiler>();
    if (shaderc->available()) {
        SPDLOG_DEBUG("using shipped shaderc as default compiler");
        cached_compiler = shaderc;
        return shaderc;
    }

    GLSLShaderCompilerHandle glslang_validator = std::make_shared<SystemGlslangValidatorCompiler>();
    if (glslang_validator->available()) {
        SPDLOG_DEBUG("using installed glslangValidator as default compiler");
        cached_compiler = glslang_validator;
        return glslang_validator;
    }

    GLSLShaderCompilerHandle glslc = std::make_shared<SystemGlslcCompiler>();
    if (glslc->available()) {
        SPDLOG_DEBUG("using installed glslc as default compiler");
        cached_compiler = glslc;
        return glslc;
    }

    SPDLOG_WARN("no shader compiler available");
    GLSLShaderCompilerHandle dummy = std::make_shared<DummyGLSLShaderCompiler>();
    cached_compiler = dummy;
    return dummy;
}

GLSLShaderCompiler::GLSLShaderCompiler() : ShaderCompiler() {}

} // namespace merian
