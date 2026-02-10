#include "merian/vk/extension/extension_glsl_compiler.hpp"
#include "merian/shader/glsl_shader_compiler_shaderc.hpp"
#include "merian/shader/glsl_shader_compiler_system_glslangValidator.hpp"
#include "merian/shader/glsl_shader_compiler_system_glslc.hpp"

#include <spdlog/spdlog.h>

namespace merian {

ExtensionGLSLCompiler::ExtensionGLSLCompiler() : ContextExtension() {}

ExtensionGLSLCompiler::~ExtensionGLSLCompiler() = default;

std::vector<std::string> ExtensionGLSLCompiler::request_extensions() {
    return {"merian-compile-context"};
}

void ExtensionGLSLCompiler::on_context_initializing(
    [[maybe_unused]] const vk::detail::DispatchLoaderDynamic& loader,
    [[maybe_unused]] const FileLoaderHandle& file_loader,
    [[maybe_unused]] const ContextCreateInfo& create_info) {

    // Try to use shaderc (preferred)
    auto shaderc = std::make_shared<ShadercCompiler>();
    if (shaderc->available()) {
        SPDLOG_DEBUG("using shipped shaderc for GLSL compilation");
        compiler = shaderc;
        return;
    }

    // Fall back to system glslangValidator
    auto glslang = std::make_shared<SystemGlslangValidatorCompiler>();
    if (glslang->available()) {
        SPDLOG_DEBUG("using system glslangValidator for GLSL compilation");
        compiler = glslang;
        return;
    }

    // Fall back to system glslc
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
    return compiler;
}

} // namespace merian
