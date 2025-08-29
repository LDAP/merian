#include "merian/vk/shader/shader_module.hpp"
#include "fullscreen_triangle.vert.spv.h"

namespace merian {

ShaderModule::ShaderModule(const ContextHandle& context,
                           const vk::ShaderModuleCreateInfo& info,
                           const vk::ArrayProxy<EntryPointInfo>& entrypoints)
    : context(context) {
    assert(!entrypoints.empty());

    shader_module = context->device.createShaderModule(info);

    for (const auto& entrypoint : entrypoints) {
        entry_points.emplace(entrypoint.get_name(), entrypoint);
    }
}

ShaderModule::~ShaderModule() {
    SPDLOG_DEBUG("destroy shader module ({})", fmt::ptr(this));
    context->device.destroyShaderModule(shader_module);
}

ShaderModule::operator const vk::ShaderModule&() const {
    return shader_module;
}

const vk::ShaderModule& ShaderModule::get_shader_module() const {
    return shader_module;
}

vk::ShaderStageFlagBits ShaderModule::get_stage_flags(const std::string& entry_point_name) const {
    const auto entry_point = entry_points.find(entry_point_name);

    if (entry_point == entry_points.end()) {
        throw std::invalid_argument{
            fmt::format("entry point {} does not exist in this module.", entry_point_name)};
    }

    return entry_point->second.get_stage_flags();
}

ShaderModule::operator ShaderStageCreateInfo() {
    return get_shader_stage_create_info();
}

ShaderStageCreateInfo
ShaderModule::get_shader_stage_create_info(const SpecializationInfoHandle& specialization_info,
                                           const char* entry_point,
                                           const vk::PipelineShaderStageCreateFlags flags) {
    if (!entry_points.contains(entry_point)) {
        throw std::invalid_argument{
            fmt::format("entry point {} does not exist in this module.", entry_point)};
    }
    return ShaderStageCreateInfo(shared_from_this(), specialization_info, entry_point, flags);
}

ShaderModuleHandle ShaderModule::fullscreen_triangle(const ContextHandle& context) {
    return create(context, merian_fullscreen_triangle_vert_spv(),
                  merian_fullscreen_triangle_vert_spv_size(),
                  ShaderModule::EntryPointInfo("main", vk::ShaderStageFlagBits::eVertex));
}

ShaderStageCreateInfo::ShaderStageCreateInfo(const ShaderModuleHandle& shader_module,
                                             const SpecializationInfoHandle& specialization_info,
                                             const std::string& entry_point,
                                             const vk::PipelineShaderStageCreateFlags flags)
    : shader_module(shader_module), specialization_info(specialization_info),
      entry_point(entry_point), flags(flags) {}

ShaderStageCreateInfo::operator vk::PipelineShaderStageCreateInfo() const {
    return get();
}

vk::PipelineShaderStageCreateInfo ShaderStageCreateInfo::operator*() const {
    return get();
}

vk::PipelineShaderStageCreateInfo ShaderStageCreateInfo::get() const {
    return vk::PipelineShaderStageCreateInfo{flags, shader_module->get_stage_flags(entry_point),
                                             *shader_module, entry_point.c_str(),
                                             *specialization_info};
}

ShaderModuleHandle ShaderModule::create(const ContextHandle& context,
                                        const vk::ShaderModuleCreateInfo& info,
                                        const vk::ArrayProxy<EntryPointInfo>& entrypoints) {
    return std::shared_ptr<ShaderModule>(new ShaderModule(context, info, entrypoints));
}

ShaderModuleHandle ShaderModule::create(const ContextHandle& context,
                                        const uint32_t spv[],
                                        const std::size_t spv_size_bytes,
                                        const vk::ArrayProxy<EntryPointInfo>& entrypoints) {
    return create(context, vk::ShaderModuleCreateInfo{{}, spv_size_bytes, spv}, entrypoints);
}

ShaderModuleHandle ShaderModule::create(const ContextHandle& context,
                                        const std::vector<uint32_t>& spv,
                                        const vk::ArrayProxy<EntryPointInfo>& entrypoints) {
    return create(context, spv.data(), spv.size() * sizeof(uint32_t), entrypoints);
}

ShaderModuleHandle ShaderModule::create(const ContextHandle& context,
                                        const void* spv,
                                        const std::size_t spv_size_bytes,
                                        const vk::ArrayProxy<EntryPointInfo>& entrypoints) {
    return create(context, (uint32_t*)spv, spv_size_bytes, entrypoints);
}

} // namespace merian
