#include "merian/vk/shader/entry_point.hpp"

#include "fullscreen_triangle.vert.spv.h"

namespace merian {

SpecializedEntryPointHandle
EntryPoint::specialize(const SpecializationInfoHandle& specialization_info) {
    return SpecializedEntryPoint::create(shared_from_this(), specialization_info);
}

SimpleEntryPointHandle SimpleEntryPoint::create(const std::string& name,
                                                const vk::ShaderStageFlagBits stage,
                                                const ShaderModuleHandle& shader_module) {
    return SimpleEntryPointHandle(new SimpleEntryPoint(name, stage, shader_module));
}

SpecializedEntryPointHandle
SpecializedEntryPoint::fullscreen_triangle(const ContextHandle& context) {
    const ShaderModuleHandle shader_module = ShaderModule::create(
        context, merian_fullscreen_triangle_vert_spv(), merian_fullscreen_triangle_vert_spv_size());
    return SimpleEntryPoint::create("main", vk::ShaderStageFlagBits::eVertex, shader_module)
        ->specialize();
}

SpecializedEntryPointHandle
SpecializedEntryPoint::create(const EntryPointHandle& entry_point,
                              const SpecializationInfoHandle& specialization_info) {
    return SpecializedEntryPointHandle(new SpecializedEntryPoint(entry_point, specialization_info));
}

SpecializedEntryPointHandle
EntryPoint::create(const std::string& name,
                   const vk::ShaderStageFlagBits stage,
                   const ShaderModuleHandle& shader_module,
                   const SpecializationInfoHandle& specialization_info) {
    return SpecializedEntryPoint::create(SimpleEntryPoint::create(name, stage, shader_module), specialization_info);
}

// shortcut to create a shader module from SPIRV and an entry point for that.
SpecializedEntryPointHandle
EntryPoint::create(const ContextHandle& context,
                   const uint32_t spv[],
                   const std::size_t spv_size,
                   const std::string& name,
                   const vk::ShaderStageFlagBits stage,
                   const SpecializationInfoHandle& specialization_info) {
    return SpecializedEntryPoint::create(
        SimpleEntryPoint::create(name, stage, ShaderModule::create(context, spv, spv_size)),
        specialization_info);
}

} // namespace merian
