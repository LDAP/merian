#pragma once

#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class EntryPoint;
using EntryPointHandle = std::shared_ptr<EntryPoint>;

class SpecializedEntryPoint;
using SpecializedEntryPointHandle = std::shared_ptr<SpecializedEntryPoint>;

class EntryPoint : std::enable_shared_from_this<EntryPoint> {

  public:
    virtual const std::string& get_name() const = 0;

    virtual vk::ShaderStageFlagBits get_stage() const = 0;

    virtual ShaderModuleHandle get_shader_module() const = 0;

    SpecializedEntryPointHandle specialize(
        const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE);

    // ----------------

    static SpecializedEntryPointHandle
    create(const std::string& name,
           const vk::ShaderStageFlagBits stage,
           const ShaderModuleHandle& shader_module,
           const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE);

    // shortcut to create a shader module from SPIRV and an entry point for that.
    static SpecializedEntryPointHandle
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
    virtual const std::string& get_name() const {
        return name;
    }

    virtual vk::ShaderStageFlagBits get_stage() const {
        return stage;
    }

    virtual ShaderModuleHandle get_shader_module() const {
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

class SpecializedEntryPoint : public EntryPoint {

  private:
    SpecializedEntryPoint(
        const EntryPointHandle& entry_point,
        const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE)
        : entry_point(entry_point), specialization_info(specialization_info) {}

  public:
    virtual const std::string& get_name() const {
        return entry_point->get_name();
    }

    virtual vk::ShaderStageFlagBits get_stage() const {
        return entry_point->get_stage();
    }

    virtual ShaderModuleHandle get_shader_module() const {
        return entry_point->get_shader_module();
    }

    // -----------------------------

    virtual SpecializationInfoHandle get_entry_point() const {
        return specialization_info;
    }

    virtual SpecializationInfoHandle get_specialization_info() const {
        return specialization_info;
    }

    virtual vk::PipelineShaderStageCreateInfo
    get_shader_stage_create_info(const vk::PipelineShaderStageCreateFlags flags = {}) const {
        return vk::PipelineShaderStageCreateInfo{flags, get_stage(), *get_shader_module(),
                                                 get_name().c_str(), *get_specialization_info()};
    }

    // ------------------------------

    // Returns a vertex shader that generates a fullscreen triangle when called with vertex count 3
    // and instance count 1.
    static SpecializedEntryPointHandle fullscreen_triangle(const ContextHandle& context);

    static SpecializedEntryPointHandle
    create(const EntryPointHandle& entry_point,
           const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE);

  private:
    const EntryPointHandle entry_point;
    const SpecializationInfoHandle specialization_info;
};

} // namespace merian
