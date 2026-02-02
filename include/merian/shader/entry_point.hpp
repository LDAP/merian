#pragma once

#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/shader/shader_module.hpp"

namespace merian {

class EntryPoint;
using EntryPointHandle = std::shared_ptr<EntryPoint>;

class VulkanEntryPoint;
using VulkanEntryPointHandle = std::shared_ptr<VulkanEntryPoint>;

/**
 * @brief      A shader entry point
 *
 * Must be able to create a shader module for Vulkan and should be possible to extend for other
 * frameworks as well.
 */
class EntryPoint : public std::enable_shared_from_this<EntryPoint> {

  public:
    virtual const char* get_name() const = 0;

    virtual vk::ShaderStageFlagBits get_stage() const = 0;

    // ----------------
    // Vulkan
    virtual ShaderModuleHandle vulkan_shader_module(const ContextHandle& context) const = 0;

    VulkanEntryPointHandle specialize(
        const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE);

    // ----------------

    static VulkanEntryPointHandle
    create(const std::string& name,
           const vk::ShaderStageFlagBits stage,
           const ShaderModuleHandle& shader_module,
           const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE);

    // shortcut to create a shader module from SPIRV and an entry point for that.
    static VulkanEntryPointHandle
    create(const ContextHandle& context,
           const uint32_t spv[],
           const std::size_t spv_size,
           const std::string& name,
           const vk::ShaderStageFlagBits stage,
           const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE);
};

class SimpleEntryPoint;
using SimpleEntryPointHandle = std::shared_ptr<SimpleEntryPoint>;

class SimpleEntryPoint : public EntryPoint {
  private:
    // use EntryPoint::create
    SimpleEntryPoint(const std::string& name,
                     const vk::ShaderStageFlagBits stage,
                     const ShaderModuleHandle& shader_module)
        : name(name), stage(stage), shader_module(shader_module) {}

  public:
    virtual const char* get_name() const override {
        return name.c_str();
    }

    virtual vk::ShaderStageFlagBits get_stage() const override {
        return stage;
    }

    virtual ShaderModuleHandle
    vulkan_shader_module(const ContextHandle& /*context*/) const override {
        return shader_module;
    }

    // --------------------------------------------

    static SimpleEntryPointHandle create(const std::string& name,
                                         const vk::ShaderStageFlagBits stage,
                                         const ShaderModuleHandle& shader_module);

  public:
  private:
    const std::string name;
    const vk::ShaderStageFlagBits stage;
    const ShaderModuleHandle shader_module;
};

class VulkanEntryPoint : public EntryPoint {

  private:
    VulkanEntryPoint(
        const EntryPointHandle& entry_point,
        const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE)
        : entry_point(entry_point), specialization_info(specialization_info) {}

  public:
    virtual const char* get_name() const {
        return entry_point->get_name();
    }

    virtual vk::ShaderStageFlagBits get_stage() const {
        return entry_point->get_stage();
    }

    virtual ShaderModuleHandle vulkan_shader_module(const ContextHandle& context) const {
        return entry_point->vulkan_shader_module(context);
    }

    // -----------------------------

    virtual EntryPointHandle get_entry_point() const {
        return entry_point;
    }

    virtual SpecializationInfoHandle get_specialization_info() const {
        return specialization_info;
    }

    virtual vk::PipelineShaderStageCreateInfo
    get_shader_stage_create_info(const ContextHandle& context,
                                 const vk::PipelineShaderStageCreateFlags flags = {}) const;

    // ------------------------------

    // Returns a vertex shader that generates a fullscreen triangle when called with vertex count 3
    // and instance count 1.
    static VulkanEntryPointHandle fullscreen_triangle(const ContextHandle& context);

    static VulkanEntryPointHandle
    create(const EntryPointHandle& entry_point,
           const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE);

  private:
    const EntryPointHandle entry_point;
    const SpecializationInfoHandle specialization_info;
};

} // namespace merian
