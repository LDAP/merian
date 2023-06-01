#include "merian/io/file_loader.hpp"
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

namespace merian {

class ShaderModuleLoader;

/**
 * @brief      Holds a vk::ShaderModule and detroys it when the object is destroyed.
 *
 * The object can only be created using the create_module(...) methods. This is to ensure there is only on object and
 * the vk::ShaderModule is destroyed when there are no references left.
 */
class ShaderModule {

  public:
    static std::shared_ptr<ShaderModule> create_module(vk::Device& device, std::string filename) {
        return std::shared_ptr<ShaderModule>(new ShaderModule(device, filename));
    }

    static std::shared_ptr<ShaderModule> create_module(vk::Device& device, vk::ShaderModuleCreateInfo& info) {
        return std::shared_ptr<ShaderModule>(new ShaderModule(device, info));
    }

    ~ShaderModule() {
        SPDLOG_DEBUG("destroy shader module from {}", filename.value_or("<unknown>"));
        device.destroyShaderModule(shaderModule);
    }

  private:
    ShaderModule() = delete;

    ShaderModule(vk::Device& device, std::string filename) : device(device), filename(filename) {
        SPDLOG_DEBUG("create shader module from {}", filename);
        std::string code = FileLoader::load_file(filename);
        vk::ShaderModuleCreateInfo info{{}, code.size(), (const uint32_t*)code.c_str()};
        shaderModule = device.createShaderModule(info);
    }

    ShaderModule(vk::Device& device, vk::ShaderModuleCreateInfo& info) : device(device), filename(std::nullopt) {
        SPDLOG_DEBUG("create shader module from {}", filename.value_or("<unknown>"));
        shaderModule = device.createShaderModule(info);
    }

  public:
    operator vk::ShaderModule&() {
        return shaderModule;
    }

    const vk::ShaderModule& get_shader_module() const {
        return shaderModule;
    }

    const std::optional<std::string>& get_filename() const {
        return filename;
    }

    vk::PipelineShaderStageCreateInfo
    get_shader_stage_create_info(vk::SpecializationInfo& spec_info,
                                 vk::ShaderStageFlagBits stage_flags = vk::ShaderStageFlagBits::eCompute,
                                 const char* entry_point = "main",
                                 vk::PipelineShaderStageCreateFlags flags = {}) {
        return vk::PipelineShaderStageCreateInfo{flags, stage_flags, shaderModule, entry_point, &spec_info};
    }

    void reload() {
        if (!filename.has_value())
            SPDLOG_WARN("requesting reload of shadermodule without filename");
        SPDLOG_DEBUG("reloading shader module {}", filename.value());
        device.destroyShaderModule(shaderModule);
        std::string code = FileLoader::load_file(filename.value());
        vk::ShaderModuleCreateInfo info{{}, code.size(), (const uint32_t*)code.c_str()};
        shaderModule = device.createShaderModule(info);
    }

  private:
    const vk::Device& device;
    const std::optional<std::string> filename;
    vk::ShaderModule shaderModule;
};

} // namespace merian
