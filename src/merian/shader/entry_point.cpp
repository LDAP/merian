#include "merian/shader/entry_point.hpp"

#include "fullscreen_triangle.vert.spv.h"

namespace merian {

VulkanEntryPointHandle
EntryPoint::specialize(const SpecializationInfoHandle& specialization_info) {
    return VulkanEntryPoint::create(shared_from_this(), specialization_info);
}

SimpleEntryPointHandle SimpleEntryPoint::create(const std::string& name,
                                                const vk::ShaderStageFlagBits stage,
                                                const ShaderModuleHandle& shader_module) {
    return SimpleEntryPointHandle(new SimpleEntryPoint(name, stage, shader_module));
}

VulkanEntryPointHandle
VulkanEntryPoint::fullscreen_triangle(const ContextHandle& context) {
    const ShaderModuleHandle shader_module = ShaderModule::create(
        context, merian_fullscreen_triangle_vert_spv(), merian_fullscreen_triangle_vert_spv_size());
    return SimpleEntryPoint::create("main", vk::ShaderStageFlagBits::eVertex, shader_module)
        ->specialize();
}

VulkanEntryPointHandle
VulkanEntryPoint::create(const EntryPointHandle& entry_point,
                              const SpecializationInfoHandle& specialization_info) {
    return VulkanEntryPointHandle(new VulkanEntryPoint(entry_point, specialization_info));
}

VulkanEntryPointHandle
EntryPoint::create(const std::string& name,
                   const vk::ShaderStageFlagBits stage,
                   const ShaderModuleHandle& shader_module,
                   const SpecializationInfoHandle& specialization_info) {
    return VulkanEntryPoint::create(SimpleEntryPoint::create(name, stage, shader_module),
                                         specialization_info);
}

// shortcut to create a shader module from SPIRV and an entry point for that.
VulkanEntryPointHandle
EntryPoint::create(const ContextHandle& context,
                   const uint32_t spv[],
                   const std::size_t spv_size,
                   const std::string& name,
                   const vk::ShaderStageFlagBits stage,
                   const SpecializationInfoHandle& specialization_info) {
    return VulkanEntryPoint::create(
        SimpleEntryPoint::create(name, stage, ShaderModule::create(context, spv, spv_size)),
        specialization_info);
}

vk::PipelineShaderStageCreateInfo
VulkanEntryPoint::get_shader_stage_create_info(const ContextHandle& context,
                                               const vk::PipelineShaderStageCreateFlags flags) const {
    return vk::PipelineShaderStageCreateInfo{flags, get_stage(), *vulkan_shader_module(context),
                                             get_name(), *get_specialization_info()};
}

} // namespace merian
