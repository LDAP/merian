#pragma once

#include "merian/io/file_loader.hpp"
#include <vulkan/vulkan.hpp>

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
                 const std::optional<vk::ShaderStageFlagBits> shader_kind = std::nullopt) {
        if (shader_kind) {
            return compile_glsl(FileLoader::load_file(path), path.string(), *shader_kind);
        } else {
            return compile_glsl(FileLoader::load_file(path), path.string(), guess_kind(path));
        }
    }

    // May throw compilation_failed.
    virtual std::vector<uint32_t> compile_glsl(const std::string& source,
                                               const std::string& source_name,
                                               const vk::ShaderStageFlagBits shader_kind) = 0;

  private:
    vk::ShaderStageFlagBits guess_kind(const std::filesystem::path& path) {
        const std::string extension = path.extension().string();
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
            throw std::runtime_error{
                fmt::format("Shader kind could not be determined for path {}", path.string())};
        }
    }
};
using ShaderCompilerHandle = std::shared_ptr<ShaderCompiler>;

} // namespace merian
