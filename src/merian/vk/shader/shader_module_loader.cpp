#include "merian/vk/shader/shader_module_loader.hpp"

namespace merian {

ShaderModuleLoader::ShaderModuleLoader(vk::Device& device, std::optional<FileLoader> file_loader)
    : device(device), file_loader(file_loader) {}

std::shared_ptr<ShaderModule> ShaderModuleLoader::load_module(std::filesystem::path path,
                                                              std::optional<FileLoader> file_loader) {

    std::optional<std::filesystem::path> full_path;
    if (file_loader.has_value())
        full_path = file_loader.value().find_file(path);
    if (!full_path.has_value() && this->file_loader.has_value()) {
        full_path = this->file_loader.value().find_file(path);
    }
    auto module = ShaderModule::create_module(device, full_path.value_or(path));
    return module;
}

} // namespace merian
