#pragma once

#include "merian/io/file_loader.hpp"
#include "merian/vk/shader/shader_module.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace merian {

class ShaderCompiler {
  public:
    class compilation_failed : public std::runtime_error {
      public:
        compilation_failed(const std::string& what) : std::runtime_error(what) {}
    };

  public:
    ShaderCompiler() {}

    virtual ~ShaderCompiler() = 0;

    // Attempt to guess the shader_kind from the file extension if shader_kind = std::nullopt.
    //
    // May throw compilation_failed.
    std::vector<uint32_t>
    compile_glsl(const std::filesystem::path& path,
                 const std::optional<vk::ShaderStageFlagBits> optional_shader_kind = std::nullopt) {
        return compile_glsl(FileLoader::load_file(path), path.string(),
                            optional_shader_kind.value_or(guess_kind(path)));
    }

    ShaderModuleHandle compile_glsl_to_shadermodule(
        const ContextHandle& context,
        const std::filesystem::path& path,
        const std::optional<vk::ShaderStageFlagBits> optional_shader_kind = std::nullopt) {
        const vk::ShaderStageFlagBits shader_kind = optional_shader_kind.value_or(guess_kind(path));
        return std::make_shared<ShaderModule>(context, compile_glsl(path, shader_kind),
                                              shader_kind);
    }

    // May throw compilation_failed.
    virtual std::vector<uint32_t> compile_glsl(const std::string& source,
                                               const std::string& source_name,
                                               const vk::ShaderStageFlagBits shader_kind) = 0;

  private:
    vk::ShaderStageFlagBits guess_kind(const std::filesystem::path& path) {
        std::string extension;
        if (path.extension().string() == ".glsl") {
            extension = std::filesystem::path(path.string().substr(0, path.string().size() - 5)).extension();
        } else {
            extension = path.extension();
        }
        if (extension == ".vert") {
            return vk::ShaderStageFlagBits::eVertex;
        } else if (extension == ".tesc") {
            return vk::ShaderStageFlagBits::eTessellationControl;
        } else if (extension == ".tese") {
            return vk::ShaderStageFlagBits::eTessellationEvaluation;
        } else if (extension == ".geom") {
            return vk::ShaderStageFlagBits::eGeometry;
        } else if (extension == ".frag") {
            return vk::ShaderStageFlagBits::eFragment;
        } else if (extension == ".comp") {
            return vk::ShaderStageFlagBits::eCompute;
        } else {
            throw compilation_failed{
                fmt::format("Shader kind could not be determined for path {}", path.string())};
        }
    }
};
using ShaderCompilerHandle = std::shared_ptr<ShaderCompiler>;

} // namespace merian
