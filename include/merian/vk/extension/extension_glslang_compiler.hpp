#pragma once

#include "merian/shader/glsl_compiler_provider.hpp"
#include "merian/shader/glsl_shader_compiler_glslang.hpp"
#include "merian/vk/extension/extension.hpp"

#include <spdlog/spdlog.h>

namespace merian {

class ExtensionGlslangCompiler : public ContextExtension, public GLSLCompilerProvider {
  public:
    static constexpr const char* name = "merian-glslang-compiler";

    DeviceSupportInfo
    query_device_support([[maybe_unused]] const DeviceSupportQueryInfo& query_info) override {
        compiler = std::make_shared<GlslangCompiler>();
        if (!compiler->available())
            return DeviceSupportInfo{false, "glslang not compiled in"};
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
