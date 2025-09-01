#pragma once

#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class EntryPoint;
using EntryPointHandle = std::shared_ptr<EntryPoint>;
class SimpleEntryPoint;
using SimpleEntryPointHandle = std::shared_ptr<SimpleEntryPoint>;

class EntryPoint {

  public:
    virtual const std::string& get_name() const = 0;

    virtual vk::ShaderStageFlagBits get_stage() const = 0;

    virtual ShaderModuleHandle get_shader_module() const = 0;

    virtual SpecializationInfoHandle get_specialization_info() const = 0;

    virtual vk::PipelineShaderStageCreateInfo
    get_shader_stage_create_info(const vk::PipelineShaderStageCreateFlags flags = {}) const {
        return vk::PipelineShaderStageCreateInfo{flags, get_stage(), *get_shader_module(),
                                                 get_name().c_str(), *get_specialization_info()};
    }

    // -----------------------------

    // Returns a new entry point that is specialized with this specialization info.
    virtual EntryPointHandle specialize(const SpecializationInfoHandle& specialization_info) = 0;

    // -----------------------------

    // Returns a vertex shader that generates a fullscreen triangle when called with vertex count 3
    // and instance count 1.
    static EntryPointHandle fullscreen_triangle(const ContextHandle& context);

    static SimpleEntryPointHandle
    create(const std::string& name,
           const vk::ShaderStageFlagBits stage,
           const ShaderModuleHandle& shader_module,
           const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE);

    // shortcut to create a shader module from SPIRV and an entry point for that.
    static SimpleEntryPointHandle
    create(const ContextHandle& context,
           const uint32_t spv[],
           const std::size_t spv_size,
           const std::string& name,
           const vk::ShaderStageFlagBits stage,
           const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE);
};

class SimpleEntryPoint : public EntryPoint {
  private:
    // use EntryPoint::create
    SimpleEntryPoint(
        const std::string& name,
        const vk::ShaderStageFlagBits stage,
        const ShaderModuleHandle& shader_module,
        const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE)
        : name(name), stage(stage), shader_module(shader_module),
          specialization_info(specialization_info) {}

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

    virtual SpecializationInfoHandle get_specialization_info() const {
        return specialization_info;
    }

    virtual EntryPointHandle specialize(const SpecializationInfoHandle& specialization_info) {
        return EntryPoint::create(name, stage, shader_module, specialization_info);
    }

    // --------------------------------------------

    static SimpleEntryPointHandle
    create(const std::string& name,
           const vk::ShaderStageFlagBits stage,
           const ShaderModuleHandle& shader_module,
           const SpecializationInfoHandle& specialization_info = MERIAN_SPECIALIZATION_INFO_NONE) {
        return SimpleEntryPointHandle(
            new SimpleEntryPoint(name, stage, shader_module, specialization_info));
    }

  public:
  private:
    const std::string name;
    const vk::ShaderStageFlagBits stage;
    const ShaderModuleHandle shader_module;
    const SpecializationInfoHandle specialization_info;
};

} // namespace merian
