#include "merian/vk/shader/shader_compiler.hpp"

#include "merian/vk/shader/glsl_shader_compiler_shaderc.hpp"
#include "merian/vk/shader/glsl_shader_compiler_system_glslangValidator.hpp"
#include "merian/vk/shader/glsl_shader_compiler_system_glslc.hpp"

namespace merian {

class DummyGLSLShaderCompiler : public GLSLShaderCompiler {
  public:
    DummyGLSLShaderCompiler(const ContextHandle& context,
                            const std::vector<std::string>& include_paths = {},
                            const std::map<std::string, std::string>& macro_definitions = {})
        : GLSLShaderCompiler(context, include_paths, macro_definitions) {}

    ~DummyGLSLShaderCompiler() {}

    std::vector<uint32_t>
    compile_glsl([[maybe_unused]] const std::string& source,
                 [[maybe_unused]] const std::string& source_name,
                 [[maybe_unused]] const vk::ShaderStageFlagBits shader_kind,
                 [[maybe_unused]] const std::vector<std::string>& additional_include_paths = {},
                 [[maybe_unused]] const std::map<std::string, std::string>&
                     additional_macro_definitions = {}) const override {
        throw compilation_failed{"compiler not available"};
    }

    bool available() const override {
        return false;
    }
};

static WeakGLSLShaderCompilerHandle cached_compiler;

GLSLShaderCompilerHandle GLSLShaderCompiler::get(const ContextHandle& context) {
    if (!cached_compiler.expired()) {
        return cached_compiler.lock();
    }

    GLSLShaderCompilerHandle shaderc = std::make_shared<ShadercCompiler>(context);
    if (shaderc->available()) {
        SPDLOG_DEBUG("using shipped shaderc as default compiler");
        cached_compiler = shaderc;
        return shaderc;
    }

    GLSLShaderCompilerHandle glslang_validator =
        std::make_shared<SystemGlslangValidatorCompiler>(context);
    if (glslang_validator->available()) {
        SPDLOG_DEBUG("using installed glslangValidator as default compiler");
        cached_compiler = glslang_validator;
        return glslang_validator;
    }

    GLSLShaderCompilerHandle glslc = std::make_shared<SystemGlslcCompiler>(context);
    if (glslc->available()) {
        SPDLOG_DEBUG("using installed glslc as default compiler");
        cached_compiler = glslc;
        return glslc;
    }

    SPDLOG_WARN("no shader compiler available");
    GLSLShaderCompilerHandle dummy = std::make_shared<DummyGLSLShaderCompiler>(context);
    cached_compiler = dummy;
    return dummy;
}

GLSLShaderCompiler::GLSLShaderCompiler(
    const ContextHandle& context,
    const std::vector<std::string>& user_include_paths,
    const std::map<std::string, std::string>& user_macro_definitions)
    : ShaderCompiler(context, user_include_paths, user_macro_definitions) {}

} // namespace merian
