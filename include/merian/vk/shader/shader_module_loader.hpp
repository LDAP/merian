#pragma once

#include "merian/io/file_loader.hpp"
#include "merian/vk/shader/shader_module.hpp"

#include <optional>
#include <vulkan/vulkan.hpp>

namespace merian {

class ShaderModuleLoader {

  public:
    ShaderModuleLoader(vk::Device& device, std::optional<FileLoader> fileLoader = std::nullopt);

    std::shared_ptr<ShaderModule> load_module(std::filesystem::path path, std::optional<FileLoader> fileLoader = std::nullopt);

  private:
    vk::Device& device;
    std::optional<FileLoader> file_loader;
};

} // namespace merian
