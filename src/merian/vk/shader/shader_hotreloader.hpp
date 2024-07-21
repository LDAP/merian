#pragma once

#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

/**
 * @brief Reloads shader modules automatically if the modified date changes.
 */
class HotReloader {
  public:
    HotReloader(const ContextHandle& context, const ShaderCompilerHandle& compiler)
        : context(context), compiler(compiler) {}

    // Compiles the shader at the specified path and returns a ShaderModule.
    // If this method is called multiple times the shader is automatically recompiled if the file
    // was changed, otherwise the same ShaderModule is returned.
    //
    // If the compilation fails, ShaderCompiler::compilation_failed might be thrown.
    ShaderModuleHandle
    get_shader(const std::filesystem::path& path,
               const std::optional<vk::ShaderStageFlagBits> shader_kind = std::nullopt);

    void clear();

  private:
    const ContextHandle context;
    const ShaderCompilerHandle compiler;

    struct per_path {
        ShaderModuleHandle shader;
        std::filesystem::file_time_type last_write_time;
        std::optional<ShaderCompiler::compilation_failed> error;
    };
    std::unordered_map<std::filesystem::path, per_path> shaders;
};
} // namespace merian
