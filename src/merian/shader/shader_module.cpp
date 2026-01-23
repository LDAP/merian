#include "merian/shader/shader_module.hpp"

namespace merian {

ShaderModule::ShaderModule(const ContextHandle& context, const vk::ShaderModuleCreateInfo& info)
    : context(context) {
    shader_module = context->get_device()->get_device().createShaderModule(info);
}

ShaderModule::ShaderModule(const ContextHandle& context,
                           const uint32_t spv[],
                           const std::size_t spv_size_bytes)
    : merian::ShaderModule(context, vk::ShaderModuleCreateInfo{{}, spv_size_bytes, spv}) {}

ShaderModule::~ShaderModule() {
    SPDLOG_DEBUG("destroy shader module ({})", fmt::ptr(this));
    context->get_device()->get_device().destroyShaderModule(shader_module);
}

ShaderModule::operator const vk::ShaderModule&() const {
    return shader_module;
}

const vk::ShaderModule& ShaderModule::get_shader_module() const {
    return shader_module;
}

ShaderModuleHandle ShaderModule::create(const ContextHandle& context,
                                        const vk::ShaderModuleCreateInfo& info) {
    return std::shared_ptr<ShaderModule>(new ShaderModule(context, info));
}

ShaderModuleHandle ShaderModule::create(const ContextHandle& context,
                                        const uint32_t spv[],
                                        const std::size_t spv_size_bytes) {
    return create(context, vk::ShaderModuleCreateInfo{{}, spv_size_bytes, spv});
}

ShaderModuleHandle ShaderModule::create(const ContextHandle& context,
                                        const std::vector<uint32_t>& spv) {
    return create(context, spv.data(), spv.size() * sizeof(uint32_t));
}

ShaderModuleHandle ShaderModule::create(const ContextHandle& context,
                                        const void* spv,
                                        const std::size_t spv_size_bytes) {
    return create(context, (uint32_t*)spv, spv_size_bytes);
}

} // namespace merian
