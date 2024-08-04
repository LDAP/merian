#include "merian/vk/shader/shader_module.hpp"
#include "fullscreen_triangle.vert.spv.h"

namespace merian {

ShaderModule::ShaderModule(const ContextHandle& context,
                           const std::string spv_filename,
                           const vk::ShaderStageFlagBits stage_flags,
                           const std::optional<FileLoader>& file_loader)
    : context(context), stage_flags(stage_flags) {
    const std::string code =
        FileLoader::load_file(file_loader.value().find_file(spv_filename).value_or(spv_filename));
    const vk::ShaderModuleCreateInfo info{{}, code.size(), (const uint32_t*)code.c_str()};
    shader_module = context->device.createShaderModule(info);
}

ShaderModule::ShaderModule(const ContextHandle& context,
                           const vk::ShaderModuleCreateInfo& info,
                           const vk::ShaderStageFlagBits stage_flags)
    : context(context), stage_flags(stage_flags) {
    shader_module = context->device.createShaderModule(info);
}

ShaderModule::ShaderModule(const ContextHandle& context,
                           const std::size_t spv_size,
                           const uint32_t spv[],
                           const vk::ShaderStageFlagBits stage_flags)
    : context(context), stage_flags(stage_flags) {
    vk::ShaderModuleCreateInfo info{{}, spv_size, spv};
    shader_module = context->device.createShaderModule(info);
}

ShaderModule::ShaderModule(const ContextHandle& context,
                           const std::vector<uint32_t>& spv,
                           const vk::ShaderStageFlagBits stage_flags)
    : ShaderModule(context, spv.size() * sizeof(uint32_t), spv.data(), stage_flags) {}

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

vk::ShaderStageFlagBits ShaderModule::get_stage_flags() const {
    return stage_flags;
}

ShaderModule::operator ShaderStageCreateInfo() {
    return get_shader_stage_create_info();
}

ShaderStageCreateInfo
ShaderModule::get_shader_stage_create_info(const SpecializationInfoHandle specialization_info,
                                           const char* entry_point,
                                           const vk::PipelineShaderStageCreateFlags flags) {
    return ShaderStageCreateInfo(shared_from_this(), specialization_info, entry_point, flags);
}

ShaderModuleHandle ShaderModule::fullscreen_triangle(const ContextHandle& context) {
    return std::make_shared<merian::ShaderModule>(
        context, merian_fullscreen_triangle_vert_spv_size(), merian_fullscreen_triangle_vert_spv(),
        vk::ShaderStageFlagBits::eVertex);
}

ShaderStageCreateInfo::ShaderStageCreateInfo(const ShaderModuleHandle& shader_module,
                                             const SpecializationInfoHandle& specialization_info,
                                             const std::string entry_point,
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
    return vk::PipelineShaderStageCreateInfo{flags, shader_module->get_stage_flags(),
                                             *shader_module, entry_point.c_str(),
                                             *specialization_info};
}

} // namespace merian
