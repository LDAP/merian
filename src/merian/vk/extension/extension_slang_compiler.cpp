#include "merian/vk/extension/extension_slang_compiler.hpp"
#include "merian/vk/extension/extension_compile_context.hpp"
#include <spdlog/spdlog.h>

namespace merian {

ExtensionSlangCompiler::ExtensionSlangCompiler() : ContextExtension() {}

ExtensionSlangCompiler::~ExtensionSlangCompiler() = default;

std::vector<std::string> ExtensionSlangCompiler::request_extensions() {
    return {"merian-compile-context"};
}

DeviceSupportInfo ExtensionSlangCompiler::query_device_support(
    [[maybe_unused]] const DeviceSupportQueryInfo& query_info) {
    // Slang is always available (embedded)
    return DeviceSupportInfo{true};
}

void ExtensionSlangCompiler::on_physical_device_selected(
    [[maybe_unused]] const PhysicalDeviceHandle& physical_device,
    const ExtensionContainer& extension_container) {
    auto compile_context_ext = extension_container.get_context_extension<ExtensionCompileContext>();
    const auto& early_compile_context = compile_context_ext->get_early_compile_context();

    // Create early session with early compile context
    early_session = SlangSession::get_or_create(early_compile_context);
}

void ExtensionSlangCompiler::on_device_created([[maybe_unused]] const DeviceHandle& device,
                                               const ExtensionContainer& extension_container) {
    auto compile_context_ext = extension_container.get_context_extension<ExtensionCompileContext>();
    const auto& compile_context = compile_context_ext->get_compile_context();

    // Create session with compile context
    session = SlangSession::get_or_create(compile_context);
}

const SlangSessionHandle& ExtensionSlangCompiler::get_early_session() const {
    if (!early_session) {
        throw MerianException("ExtensionSlangCompiler: early session not available yet "
                              "(called before physical device selection)");
    }
    return early_session;
}

const SlangSessionHandle& ExtensionSlangCompiler::get_session() const {
    if (!session) {
        throw MerianException("ExtensionSlangCompiler: session not available yet "
                              "(called before device creation)");
    }
    return session;
}

bool ExtensionSlangCompiler::has_early_session() const {
    return early_session != nullptr;
}

bool ExtensionSlangCompiler::has_session() const {
    return session != nullptr;
}

} // namespace merian
