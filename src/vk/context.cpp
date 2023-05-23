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

bool check_layer_support(std::vector<const char*> layers) {
    std::vector<vk::LayerProperties> layer_props = vk::enumerateInstanceLayerProperties();

    for (auto& layer : layers) {
        bool found = false;
        std::string layer_str = layer;
        for (auto& layer_prop : layer_props) {
            if (layer_prop.layerName.data() == layer_str) {
                found = true;
                break;
            }
        }
        if (!found)
            throw std::runtime_error(fmt::format("layer '{}' not supported", layer));
    }

    return true;
}

Context::Context() {
    // Init dynamic loader
    static vk::DynamicLoader dl;
    static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

#ifdef DEBUG
    extensions.push_back(new ExtensionDebugUtils());
#endif
    extensions.push_back(new ExtensionGLFW());
    spdlog::debug("Active extensions:");
    for (auto& ext : extensions) {
        spdlog::debug("{}", ext->name());
    }

    create_instance();
    // Must happen before on_instance_created since it requires dynamic loading
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
    for (auto& ext : extensions) {
        ext->on_instance_created(instance);
    }
}

Context::~Context() {
    for (auto& ext : extensions) {
        spdlog::debug("destroy extension {}", ext->name());
        ext->on_destroy(instance);

        delete ext;
    }
    spdlog::debug("destroy instance");
    instance.destroy();
}

void Context::create_instance() {
    std::vector<const char*> layer_names;
    std::vector<const char*> extension_names;
    for (auto& ext : extensions) {
        insert_all(layer_names, ext->required_layer_names());
        insert_all(extension_names, ext->required_extension_names());
    }

    spdlog::debug("requested layers: [{}]", fmt::join(layer_names, ", "));
    spdlog::debug("requested extensions: [{}]", fmt::join(extension_names, ", "));

    check_layer_support(layer_names);

    void* p_next = nullptr;
    for (auto& ext : extensions) {
        p_next = ext->on_create_instance(p_next);
    }

    vk::InstanceCreateInfo instance_create_info{
        {},
        &application_info,
        static_cast<uint32_t>(layer_names.size()),
        layer_names.data(),
        static_cast<uint32_t>(extension_names.size()),
        extension_names.data(),
        p_next,
    };

    instance = vk::createInstance(instance_create_info);
    spdlog::debug("instance created");
}
