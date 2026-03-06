#include "merian/vk/extension/extension_glsl_compiler.hpp"
#include "merian/shader/glsl_shader_compiler_glslang.hpp"
#include "merian/shader/glsl_shader_compiler_shaderc.hpp"
#include "merian/shader/glsl_shader_compiler_system_glslangValidator.hpp"
#include "merian/shader/glsl_shader_compiler_system_glslc.hpp"

#include <spdlog/spdlog.h>

namespace merian {

ExtensionGLSLCompiler::ExtensionGLSLCompiler() : ContextExtension() {}

ExtensionGLSLCompiler::~ExtensionGLSLCompiler() = default;

void ExtensionGLSLCompiler::on_context_initializing(
    [[maybe_unused]] const PFN_vkGetInstanceProcAddr loader,
    [[maybe_unused]] const FileLoaderHandle& file_loader,
    [[maybe_unused]] const ContextCreateInfo& create_info) {

    // Note: glslang and glslangValidator provide the best experience in debugging and profiling
    // tools.

    // Try linked glslang
    auto glslang = std::make_shared<GlslangCompiler>();
    if (glslang->available()) {
        SPDLOG_DEBUG("using linked glslang for GLSL compilation");
        compiler = glslang;
        return;
    }

    // Try system glslangValidator
    auto glslang_validator = std::make_shared<SystemGlslangValidatorCompiler>();
    if (glslang_validator->available()) {
        SPDLOG_DEBUG("using system glslangValidator for GLSL compilation");
        compiler = glslang_validator;
        return;
    }

    // Try bundled shaderc
    auto shaderc = std::make_shared<ShadercCompiler>();
    if (shaderc->available()) {
        SPDLOG_DEBUG("using linked shaderc for GLSL compilation");
        compiler = shaderc;
        return;
    }

    // Try system glslc
    auto glslc = std::make_shared<SystemGlslcCompiler>();
    if (glslc->available()) {
        SPDLOG_DEBUG("using system glslc for GLSL compilation");
        compiler = glslc;
        return;
    }

    // No compiler available - this should not happen as query_device_support should have failed
    SPDLOG_ERROR("no GLSL compiler available");
    throw MerianException("GLSL compiler not available");
}

DeviceSupportInfo ExtensionGLSLCompiler::query_device_support(
    [[maybe_unused]] const DeviceSupportQueryInfo& query_info) {

    if (compiler != nullptr) {
        return DeviceSupportInfo{true};
    }

    return DeviceSupportInfo{compiler != nullptr, "no GLSL compiler available"};
}

const GLSLShaderCompilerHandle& ExtensionGLSLCompiler::get_compiler() const {
    assert(compiler != nullptr);
    return compiler;
}

} // namespace merian
