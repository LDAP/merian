#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/shader/shader_compile_context.hpp"

namespace merian {

/**
 * @brief Extension that provides shader compilation contexts.
 *
 * Manages two ShaderCompileContexts:
 * - early_compile_context: Available after physical device selection (uses physical device defines)
 * - compile_context: Available after device creation (uses device defines)
 */
class ExtensionCompileContext : public ContextExtension {
  public:
    ExtensionCompileContext();

    ~ExtensionCompileContext() override;

    void on_context_initializing(const vk::detail::DispatchLoaderDynamic& loader,
                                 const FileLoader& file_loader,
                                 const ContextCreateInfo& create_info) override;

    void on_physical_device_selected(const PhysicalDeviceHandle& physical_device,
                                     const ExtensionContainer& extension_container) override;

    void on_device_created(const DeviceHandle& device,
                          const ExtensionContainer& extension_container) override;

    const ShaderCompileContextHandle& get_early_compile_context() const;

    const ShaderCompileContextHandle& get_compile_context() const;

    bool has_early_compile_context() const;

    bool has_compile_context() const;

  private:
    FileLoader stored_file_loader;
    ShaderCompileContextHandle early_compile_context;
    ShaderCompileContextHandle compile_context;
};

} // namespace merian
