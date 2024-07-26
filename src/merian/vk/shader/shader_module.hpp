#pragma once

#include "merian/io/file_loader.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <optional>

namespace merian {

class ShaderStageCreateInfo;

/**
 * @brief      Holds a vk::ShaderModule and detroys it when the object is destroyed.
 *
 * The object can only be created using the create_module(...) methods. This is to ensure there is
 * only on object and the vk::ShaderModule is destroyed when there are no references left.
 */
class ShaderModule : public std::enable_shared_from_this<ShaderModule> {
  public:
    ShaderModule() = delete;

    ShaderModule(const ContextHandle& context,
                 const std::string spv_filename,
                 const vk::ShaderStageFlagBits stage_flags = vk::ShaderStageFlagBits::eCompute,
                 const std::optional<FileLoader>& file_loader = std::nullopt)
        : context(context), stage_flags(stage_flags) {
        const std::string code = FileLoader::load_file(
            file_loader.value().find_file(spv_filename).value_or(spv_filename));
        const vk::ShaderModuleCreateInfo info{{}, code.size(), (const uint32_t*)code.c_str()};
        shader_module = context->device.createShaderModule(info);
    }

    ShaderModule(const ContextHandle& context,
                 const vk::ShaderModuleCreateInfo& info,
                 const vk::ShaderStageFlagBits stage_flags = vk::ShaderStageFlagBits::eCompute)
        : context(context), stage_flags(stage_flags) {
        shader_module = context->device.createShaderModule(info);
    }

    ShaderModule(const ContextHandle& context,
                 const std::size_t spv_size,
                 const uint32_t spv[],
                 const vk::ShaderStageFlagBits stage_flags = vk::ShaderStageFlagBits::eCompute)
        : context(context), stage_flags(stage_flags) {
        vk::ShaderModuleCreateInfo info{{}, spv_size, spv};
        shader_module = context->device.createShaderModule(info);
    }

    ShaderModule(const ContextHandle& context,
                 const std::vector<uint32_t>& spv,
                 const vk::ShaderStageFlagBits stage_flags = vk::ShaderStageFlagBits::eCompute)
        : ShaderModule(context, spv.size() * sizeof(uint32_t), spv.data(), stage_flags) {}

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

    vk::ShaderStageFlagBits get_stage_flags() const {
        return stage_flags;
    }

    operator ShaderStageCreateInfo();

    ShaderStageCreateInfo get_shader_stage_create_info(
        const SpecializationInfoHandle specialization_info = MERIAN_SPECIALIZATION_INFO_NONE,
        const char* entry_point = "main",
        const vk::PipelineShaderStageCreateFlags flags = {});

  private:
    const ContextHandle context;
    const vk::ShaderStageFlagBits stage_flags;

    vk::ShaderModule shader_module;
};

using ShaderModuleHandle = std::shared_ptr<ShaderModule>;

class ShaderStageCreateInfo {
  public:
    ShaderStageCreateInfo(
        const ShaderModuleHandle& shader_module,
        const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE,
        const std::string entry_point = "main",
        const vk::PipelineShaderStageCreateFlags flags = {})
        : shader_module(shader_module), specialization_info(specialization_info),
          entry_point(entry_point), flags(flags) {}

    operator vk::PipelineShaderStageCreateInfo() const {
        return get();
    }

    vk::PipelineShaderStageCreateInfo operator*() const {
        return get();
    }

    vk::PipelineShaderStageCreateInfo get() const {
        return vk::PipelineShaderStageCreateInfo{flags, shader_module->get_stage_flags(),
                                                 *shader_module, entry_point.c_str(),
                                                 *specialization_info};
    }

    const ShaderModuleHandle shader_module;
    const SpecializationInfoHandle specialization_info;
    const std::string entry_point;
    const vk::PipelineShaderStageCreateFlags flags;
};
using ShaderStageCreateInfoHandle = std::shared_ptr<ShaderStageCreateInfo>;

inline ShaderModule::operator ShaderStageCreateInfo() {
    return get_shader_stage_create_info();
}

inline ShaderStageCreateInfo
ShaderModule::get_shader_stage_create_info(const SpecializationInfoHandle specialization_info,
                                           const char* entry_point,
                                           const vk::PipelineShaderStageCreateFlags flags) {
    return ShaderStageCreateInfo(shared_from_this(), specialization_info, entry_point, flags);
}

} // namespace merian
