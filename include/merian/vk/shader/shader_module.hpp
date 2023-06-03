#pragma once

#include "merian/io/file_loader.hpp"
#include "merian/vk/context.hpp"
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

namespace merian {

class ShaderModuleLoader;

/**
 * @brief      Holds a vk::ShaderModule and detroys it when the object is destroyed.
 *
 * The object can only be created using the create_module(...) methods. This is to ensure there is
 * only on object and the vk::ShaderModule is destroyed when there are no references left.
 */
class ShaderModule {

  public:
    ShaderModule() = delete;

    ShaderModule(const SharedContext& context,
                 const std::string filename,
                 const std::optional<FileLoader> file_loader = std::nullopt)
        : context(context) {
        if (file_loader.has_value())
            this->filename = file_loader.value().find_file(filename).value_or(filename);

        std::string code = FileLoader::load_file(this->filename.value());
        vk::ShaderModuleCreateInfo info{{}, code.size(), (const uint32_t*)code.c_str()};
        create_module(info);
    }

    ShaderModule(const SharedContext& context, const vk::ShaderModuleCreateInfo& info)
        : context(context), filename(std::nullopt) {
        create_module(info);
    }

    ~ShaderModule() {
        SPDLOG_DEBUG("destroy shader module from {} ({})", filename.value_or("<unknown>"),
                     fmt::ptr(this));
        context->device.destroyShaderModule(shader_module);
    }

  public:
    operator vk::ShaderModule&() {
        return shader_module;
    }

    const vk::ShaderModule& get_shader_module() const {
        return shader_module;
    }

    const std::optional<std::string>& get_filename() const {
        return filename;
    }

    vk::PipelineShaderStageCreateInfo get_shader_stage_create_info(
        const vk::ShaderStageFlagBits stage_flags = vk::ShaderStageFlagBits::eCompute,
        const vk::SpecializationInfo& spec_info = {},
        const char* entry_point = "main",
        const vk::PipelineShaderStageCreateFlags flags = {}) {
        return vk::PipelineShaderStageCreateInfo{flags, stage_flags, shader_module, entry_point,
                                                 &spec_info};
    }

    void reload() {
        if (!filename.has_value())
            SPDLOG_WARN("requesting reload of shadermodule without filename");
        SPDLOG_DEBUG("reloading shader module {}", filename.value());
        context->device.destroyShaderModule(shader_module);
        std::string code = FileLoader::load_file(filename.value());
        vk::ShaderModuleCreateInfo info{{}, code.size(), (const uint32_t*)code.c_str()};
        shader_module = context->device.createShaderModule(info);
    }

  private:
    void create_module(const vk::ShaderModuleCreateInfo info) {
        SPDLOG_DEBUG("create shader module from {} ({})", filename.value_or("<unknown>"),
                     fmt::ptr(this));
        shader_module = context->device.createShaderModule(info);
    }

  private:
    const SharedContext context;
    std::optional<std::string> filename;
    vk::ShaderModule shader_module;
};

} // namespace merian
