#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/object.hpp"

namespace merian {

class ShaderModule;
using ShaderModuleHandle = std::shared_ptr<ShaderModule>;

/**
 * @brief      Holds a vk::ShaderModule and detroys it when the object is destroyed.
 *
 * The object can only be created using the create_module(...) methods. This is to ensure there is
 * only on object and the vk::ShaderModule is destroyed when there are no references left.
 */
class ShaderModule : public std::enable_shared_from_this<ShaderModule>, public Object {
  public:
    ShaderModule() = delete;

  private:
    ShaderModule(const ContextHandle& context, const vk::ShaderModuleCreateInfo& info);

  public:
    ~ShaderModule();

    operator const vk::ShaderModule&() const;

    const vk::ShaderModule& get_shader_module() const;

  public:
    static ShaderModuleHandle create(const ContextHandle& context,
                                     const vk::ShaderModuleCreateInfo& info);

    static ShaderModuleHandle
    create(const ContextHandle& context, const uint32_t spv[], const std::size_t spv_size);

    static ShaderModuleHandle create(const ContextHandle& context,
                                     const std::vector<uint32_t>& spv);

    static ShaderModuleHandle
    create(const ContextHandle& context, const void* spv, const std::size_t spv_size);

  private:
    const ContextHandle context;
    vk::ShaderModule shader_module;
};

} // namespace merian
