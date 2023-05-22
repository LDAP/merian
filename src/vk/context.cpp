#include "utils/vector_utils.hpp"
#include "vk/extension/extension_debug_utils.hpp"
#include "vk/extension/extension_glfw.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <vk/context.hpp>
#include <vk/extension/extension.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

static const vk::ApplicationInfo application_info{
    PROJECT_NAME,       VK_MAKE_VERSION(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH),
    PROJECT_NAME,       VK_MAKE_VERSION(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH),
    VK_API_VERSION_1_3,
};

static std::vector<Extension*> extensions{
#ifdef DEBUG
    new ExtensionDebugUtils(),
#endif
    new ExtensionGLFW(),
};

Context::Context() {
    // Init dynamic loader
    static vk::DynamicLoader dl;
    static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    spdlog::debug("Active extensions:");
    for (auto& ext : extensions) {
        spdlog::debug("{}", ext->name());
    }

    std::vector<const char*> layer_names;
    std::vector<const char*> extension_names;
    for (auto& ext : extensions) {
        insert_all(layer_names, ext->required_layer_names());
        insert_all(extension_names, ext->required_extension_names());
    }

    vk::InstanceCreateInfo instance_create_info{
        {},
        &application_info,
        static_cast<uint32_t>(layer_names.size()),
        layer_names.data(),
        static_cast<uint32_t>(extension_names.size()),
        extension_names.data(),
    };

    instance = vk::createInstance(instance_create_info);
    for (auto& ext: extensions) {
        ext->on_instance_created(instance);
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
}
