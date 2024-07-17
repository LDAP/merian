#pragma once

#include "merian/io/file_loader.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <optional>

namespace merian {

class ShaderModuleLoader;

/**
 * @brief      Holds a vk::ShaderModule and detroys it when the object is destroyed.
 *
 * The object can only be created using the create_module(...) methods. This is to ensure there is
 * only on object and the vk::ShaderModule is destroyed when there are no references left.
 */
class ShaderModule : public std::enable_shared_from_this<ShaderModule> {
  public:
  public:
    ShaderModule() = delete;

    ShaderModule(const ContextHandle& context,
                 const std::string spv_filename,
                 const std::optional<FileLoader>& file_loader = std::nullopt)
        : context(context) {
        std::string code = FileLoader::load_file(
            file_loader.value().find_file(spv_filename).value_or(spv_filename));
        vk::ShaderModuleCreateInfo info{{}, code.size(), (const uint32_t*)code.c_str()};
        shader_module = context->device.createShaderModule(info);
    }

    ShaderModule(const ContextHandle& context, const vk::ShaderModuleCreateInfo& info)
        : context(context) {
        shader_module = context->device.createShaderModule(info);
    }

    ShaderModule(const ContextHandle& context, const std::size_t spv_size, const uint32_t spv[])
        : context(context) {
        vk::ShaderModuleCreateInfo info{{}, spv_size, spv};
        shader_module = context->device.createShaderModule(info);
    }

    ShaderModule(const ContextHandle& context, const std::vector<uint32_t>& spv)
        : ShaderModule(context, spv.size() * sizeof(uint32_t), spv.data()) {}

    ~ShaderModule() {
        SPDLOG_DEBUG("destroy shader module ({})", fmt::ptr(this));
        context->device.destroyShaderModule(shader_module);
    }

  public:
    operator const vk::ShaderModule&() const {
        return shader_module;
    }

    const vk::ShaderModule& get_shader_module() const {
        return shader_module;
    }

    vk::PipelineShaderStageCreateInfo get_shader_stage_create_info(
        const vk::ShaderStageFlagBits stage_flags = vk::ShaderStageFlagBits::eCompute,
        const SpecializationInfoHandle specialization_info = MERIAN_SPECIALIZATION_INFO_NONE,
        const char* entry_point = "main",
        const vk::PipelineShaderStageCreateFlags flags = {}) {
        return vk::PipelineShaderStageCreateInfo{flags, stage_flags, shader_module, entry_point,
                                                 *specialization_info};
    }

  private:
    const ContextHandle context;
    vk::ShaderModule shader_module;
};

using ShaderModuleHandle = std::shared_ptr<ShaderModule>;

} // namespace merian
