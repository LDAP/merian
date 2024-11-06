#pragma once

#include "merian/io/file_loader.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"

#include <optional>

namespace merian {

class ShaderModule;
using ShaderModuleHandle = std::shared_ptr<ShaderModule>;
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
                 const std::string& spv_filename,
                 const vk::ShaderStageFlagBits stage_flags = vk::ShaderStageFlagBits::eCompute,
                 const std::optional<FileLoader>& file_loader = std::nullopt);

    ShaderModule(const ContextHandle& context,
                 const vk::ShaderModuleCreateInfo& info,
                 const vk::ShaderStageFlagBits stage_flags = vk::ShaderStageFlagBits::eCompute);
    ShaderModule(const ContextHandle& context,
                 const std::size_t spv_size,
                 const uint32_t spv[],
                 const vk::ShaderStageFlagBits stage_flags = vk::ShaderStageFlagBits::eCompute);

    ShaderModule(const ContextHandle& context,
                 const std::vector<uint32_t>& spv,
                 const vk::ShaderStageFlagBits stage_flags = vk::ShaderStageFlagBits::eCompute);

    ~ShaderModule();

  public:
    operator const vk::ShaderModule&() const;

    const vk::ShaderModule& get_shader_module() const;

    vk::ShaderStageFlagBits get_stage_flags() const;

    operator ShaderStageCreateInfo();

    ShaderStageCreateInfo get_shader_stage_create_info(
        const SpecializationInfoHandle specialization_info = MERIAN_SPECIALIZATION_INFO_NONE,
        const char* entry_point = "main",
        const vk::PipelineShaderStageCreateFlags flags = {});

  public:
    // Returns a vertex shader that generates a fullscreen triangle when called with vertex count 3
    // and instance count 1.
    static ShaderModuleHandle fullscreen_triangle(const ContextHandle& context);

  private:
    const ContextHandle context;
    const vk::ShaderStageFlagBits stage_flags;

    vk::ShaderModule shader_module;
};

class ShaderStageCreateInfo {
  public:
    ShaderStageCreateInfo(
        const ShaderModuleHandle& shader_module,
        const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE,
        const std::string entry_point = "main",
        const vk::PipelineShaderStageCreateFlags flags = {});

    operator vk::PipelineShaderStageCreateInfo() const;

    vk::PipelineShaderStageCreateInfo operator*() const;

    vk::PipelineShaderStageCreateInfo get() const;

    const ShaderModuleHandle shader_module;
    const SpecializationInfoHandle specialization_info;
    const std::string entry_point;
    const vk::PipelineShaderStageCreateFlags flags;
};
using ShaderStageCreateInfoHandle = std::shared_ptr<ShaderStageCreateInfo>;

} // namespace merian
