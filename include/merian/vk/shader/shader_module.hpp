#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/object.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"

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
class ShaderModule : public std::enable_shared_from_this<ShaderModule>, public Object {
  public:
    class EntryPointInfo {
      public:
        EntryPointInfo(const std::string& name, const vk::ShaderStageFlagBits stage_flags)
            : name(name), stage_flags(stage_flags) {}

        const std::string& get_name() const {
            return name;
        }

        vk::ShaderStageFlagBits get_stage_flags() const {
            return stage_flags;
        }

      private:
        const std::string name;
        const vk::ShaderStageFlagBits stage_flags;
    };

  public:
    ShaderModule() = delete;

  private:
    ShaderModule(const ContextHandle& context,
                 const vk::ShaderModuleCreateInfo& info,
                 const vk::ArrayProxy<EntryPointInfo>& entrypoints);

  public:
    ~ShaderModule();

    operator const vk::ShaderModule&() const;

    const vk::ShaderModule& get_shader_module() const;

    // may throw invalid argument if the entry point does not exist
    vk::ShaderStageFlagBits get_stage_flags(const std::string& entry_point_name) const;

    operator ShaderStageCreateInfo();

    ShaderStageCreateInfo get_shader_stage_create_info(
        const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE,
        const char* entry_point = "main",
        const vk::PipelineShaderStageCreateFlags flags = {});

  public:
    // Returns a vertex shader that generates a fullscreen triangle when called with vertex count 3
    // and instance count 1.
    static ShaderModuleHandle fullscreen_triangle(const ContextHandle& context);

    static ShaderModuleHandle create(const ContextHandle& context,
                                     const vk::ShaderModuleCreateInfo& info,
                                     const vk::ArrayProxy<EntryPointInfo>& entrypoints);

    static ShaderModuleHandle create(const ContextHandle& context,
                                     const uint32_t spv[],
                                     const std::size_t spv_size,
                                     const vk::ArrayProxy<EntryPointInfo>& entrypoints);

    static ShaderModuleHandle create(const ContextHandle& context,
                                     const std::vector<uint32_t>& spv,
                                     const vk::ArrayProxy<EntryPointInfo>& entrypoints);

    static ShaderModuleHandle create(const ContextHandle& context,
                                     const void* spv,
                                     const std::size_t spv_size,
                                     const vk::ArrayProxy<EntryPointInfo>& entrypoints);

  private:
    const ContextHandle context;
    std::map<std::string, EntryPointInfo> entry_points;

    vk::ShaderModule shader_module;
};

class ShaderStageCreateInfo {
  public:
    ShaderStageCreateInfo(
        const ShaderModuleHandle& shader_module,
        const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE,
        const std::string& entry_point = "main",
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
