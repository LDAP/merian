#pragma once

#include "merian/shader/glsl_compiler_provider.hpp"
#include "merian/shader/glsl_shader_compiler_system_glslangValidator.hpp"
#include "merian/vk/extension/extension.hpp"

#include <spdlog/spdlog.h>

namespace merian {

class ExtensionGlslangValidatorCompiler : public ContextExtension, public GLSLCompilerProvider {
  public:
    static constexpr const char* name = "merian-glslangvalidator-compiler";

    DeviceSupportInfo
    query_device_support([[maybe_unused]] const DeviceSupportQueryInfo& query_info) override {
        compiler = std::make_shared<SystemGlslangValidatorCompiler>();
        if (!compiler->available())
            return DeviceSupportInfo{false, "glslangValidator not found in PATH"};
        return DeviceSupportInfo{true};
    }

    void on_unsupported([[maybe_unused]] const std::string& reason) override {
        compiler.reset();
        SPDLOG_DEBUG("extension {} not supported ({})", name, reason);
    }

    GLSLShaderCompilerHandle get_glsl_compiler() const override {
        return compiler;
    }

  private:
    GLSLShaderCompilerHandle compiler;
};

} // namespace merian
