#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/*
 * Extension that enables the Vulkan features required by ImGuiRenderer:
 *   - dynamicRendering feature (Vulkan 1.3 core)
 *   - VK_KHR_push_descriptor device extension
 */
class ExtensionImGui : public ContextExtension {
  public:
    static constexpr const char* name = "merian-imgui";

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override {
        return DeviceSupportInfo::check(query_info, {"dynamicRendering", "pushDescriptor"});
    }
};

using ExtensionImGuiHandle = std::shared_ptr<ExtensionImGui>;

} // namespace merian
