#pragma once

#include "merian/shader/glsl_shader_compiler.hpp"
#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * @brief Extension that provides GLSL shader compilation services.
 *
 * Depends on merian-compile-context extension.
 * Returns unsupported if no GLSL compiler is available.
 */
class ExtensionGLSLCompiler : public ContextExtension {
  public:
    ExtensionGLSLCompiler();

    ~ExtensionGLSLCompiler() override;

    std::vector<std::string> request_extensions() override;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void on_context_initializing(const vk::detail::DispatchLoaderDynamic& loader,
                                 const FileLoaderHandle& file_loader,
                                 const ContextCreateInfo& create_info) override;

    const GLSLShaderCompilerHandle& get_compiler() const;

  private:
    GLSLShaderCompilerHandle compiler;
};

} // namespace merian
