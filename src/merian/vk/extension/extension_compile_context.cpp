#include "merian/vk/extension/extension_compile_context.hpp"
#include <spdlog/spdlog.h>

namespace merian {

ExtensionCompileContext::ExtensionCompileContext() : ContextExtension() {}

ExtensionCompileContext::~ExtensionCompileContext() = default;

void ExtensionCompileContext::on_context_initializing(
    [[maybe_unused]] const vk::detail::DispatchLoaderDynamic& loader,
    const FileLoaderHandle& file_loader,
    [[maybe_unused]] const ContextCreateInfo& create_info) {
    this->file_loader = file_loader;
}

void ExtensionCompileContext::on_physical_device_selected(
    const PhysicalDeviceHandle& physical_device,
    [[maybe_unused]] const ExtensionContainer& extension_container) {
    // Create early compile context with physical device defines
    early_compile_context = ShaderCompileContext::create(*file_loader, physical_device);
}

void ExtensionCompileContext::on_device_created(
    const DeviceHandle& device, [[maybe_unused]] const ExtensionContainer& extension_container) {
    // Create compile context with device defines
    compile_context = ShaderCompileContext::create(*file_loader, device);
}

const ShaderCompileContextHandle& ExtensionCompileContext::get_early_compile_context() const {
    if (!early_compile_context) {
        throw MerianException("ExtensionCompileContext: early compile context not available yet "
                              "(called before physical device selection)");
    }
    return early_compile_context;
}

const ShaderCompileContextHandle& ExtensionCompileContext::get_compile_context() const {
    if (!compile_context) {
        throw MerianException("ExtensionCompileContext: compile context not available yet "
                              "(called before device creation)");
    }
    return compile_context;
}

bool ExtensionCompileContext::has_early_compile_context() const {
    return early_compile_context != nullptr;
}

bool ExtensionCompileContext::has_compile_context() const {
    return compile_context != nullptr;
}

} // namespace merian
