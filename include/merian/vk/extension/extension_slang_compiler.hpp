#pragma once

#include "merian/shader/slang_session.hpp"
#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * @brief Extension that provides Slang shader compilation services.
 *
 * Depends on merian-compile-context extension.
 * Manages two Slang sessions:
 * - early_session: Available after physical device selection, uses physical device defines
 * - session: Available after device creation, uses device defines (more complete)
 */
class ExtensionSlangCompiler : public ContextExtension {
  public:
    ExtensionSlangCompiler();

    ~ExtensionSlangCompiler() override;

    std::vector<std::string> request_extensions() override;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void on_physical_device_selected(const PhysicalDeviceHandle& physical_device,
                                     const ExtensionContainer& extension_container) override;

    void on_device_created(const DeviceHandle& device,
                           const ExtensionContainer& extension_container) override;

    const SlangSessionHandle& get_early_session() const;

    const SlangSessionHandle& get_session() const;

    bool has_early_session() const;

    bool has_session() const;

  private:
    SlangSessionHandle early_session;
    SlangSessionHandle session;
};

} // namespace merian
