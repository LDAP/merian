#include "utils/vector_utils.hpp"
#include "vk/extension/extension_debug_utils.hpp"
#include "vk/extension/extension_float_atomics.hpp"
#include "vk/extension/extension_glfw.hpp"
#include "vk/extension/extension_raytrace.hpp"
#include "vk/extension/extension_v12.hpp"

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

Context::Context(uint32_t vendor_id, uint32_t device_id) {
    // Init dynamic loader
    static vk::DynamicLoader dl;
    static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

#ifdef DEBUG
    extensions.push_back(new ExtensionDebugUtils());
#endif
    extensions.push_back(new ExtensionGLFW());
    extensions.push_back(new ExtensionRaytraceQuery());
    extensions.push_back(new ExtensionV12());
    extensions.push_back(new ExtensionFloatAtomics());

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
    prepare_physical_device(vendor_id, device_id);
    find_queues();
    create_device_and_queues();
    create_command_pools();

    // for (auto& ext : extensions) {
    //     ext->on_context_created(*this);
    // }
}

void destroy_extensions(Context& context, std::vector<Extension*> extensions) {
    for (auto& ext : extensions) {
        spdlog::debug("destroy extension {}", ext->name());
        ext->on_destroy(context.instance);
        delete ext;

        for (std::size_t i = 0; context.extensions.size(); i++) {
            if (context.extensions[i] == ext) {
                std::swap(context.extensions[i], context.extensions[context.extensions.size() - 1]);
                context.extensions.pop_back();
                break;
            }
        }
    }
}

Context::~Context() {
    spdlog::debug("destroy command pools");
    device.destroyCommandPool(cmd_pool_graphics);
    device.destroyCommandPool(cmd_pool_transfer);

    spdlog::debug("destroy device");
    device.destroy();

    destroy_extensions(*this, extensions);

    spdlog::debug("destroy instance");
    instance.destroy();
}

void Context::create_instance() {
    for (auto& ext : extensions) {
        insert_all(layer_names, ext->required_layer_names());
        insert_all(instance_extension_names, ext->required_instance_extension_names());
    }

    spdlog::debug("required layers: [{}]", fmt::join(layer_names, ", "));
    spdlog::debug("required instance extensions: [{}]", fmt::join(instance_extension_names, ", "));

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
        static_cast<uint32_t>(instance_extension_names.size()),
        instance_extension_names.data(),
        p_next,
    };

    instance = vk::createInstance(instance_create_info);
    spdlog::debug("instance created");
}

void Context::prepare_physical_device(uint32_t vendor_id, uint32_t device_id) {
    std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No vulkan device found!");
    }

    long selected = -1;
    for (std::size_t i = 0; i < devices.size(); i++) {
        vk::PhysicalDeviceProperties2 props = devices[i].getProperties2();
        spdlog::info("Found physical device {}, vendor id: {}, device id: {}", props.properties.deviceName,
                     props.properties.vendorID, props.properties.deviceID);
        if ((vendor_id == (uint32_t)-1 || vendor_id == props.properties.vendorID) &&
            (device_id == (uint32_t)-1 || device_id == props.properties.deviceID)) {
            selected = i;
        }
    }

    if (selected == -1) {
        throw std::runtime_error(fmt::format("No vulkan device found with vendor id: {}, device id: {}! (-1 means any)",
                                             vendor_id, device_id));
    }
    physical_device = devices[selected];
    physical_device_props = physical_device.getProperties();
    spdlog::info("Selected physical device {}, vendor id: {}, device id: {}",
                 physical_device_props.properties.deviceName, physical_device_props.properties.vendorID,
                 physical_device_props.properties.deviceID);

    physical_device_features = physical_device.getFeatures2();
    physical_device_memory_properties = physical_device.getMemoryProperties2();
    extension_properties = physical_device.enumerateDeviceExtensionProperties();

    spdlog::debug("Checking extension support...");
    std::vector<Extension*> not_supported;
    for (auto& ext : extensions) {
        if (!ext->extension_supported(physical_device, extension_properties)) {
            spdlog::info("Extension {} not supported, disabling...", ext->name());
            not_supported.push_back(ext);
        } else {
            spdlog::debug("Extension {} supported", ext->name());
        }
    }
    destroy_extensions(*this, not_supported);
}

void Context::find_queues() {
    std::vector<vk::QueueFamilyProperties> queue_family_props = physical_device.getQueueFamilyProperties();
    int queue_idx_graphics = -1;
    int queue_idx_transfer = -1;

    for (std::size_t i = 0; i < queue_family_props.size(); i++) {
        if (!queue_family_props[i].queueCount)
            continue;

        const bool supports_graphics = queue_family_props[i].queueFlags & vk::QueueFlagBits::eGraphics ? true : false;
        const int supports_compute = queue_family_props[i].queueFlags & vk::QueueFlagBits::eCompute ? true : false;
        const int supports_transfer = queue_family_props[i].queueFlags & vk::QueueFlagBits::eTransfer ? true : false;

        if (queue_idx_graphics < 0 && supports_graphics && supports_compute) {
            bool accept = true;
            for (auto& ext : extensions) {
                if (!ext->accept_graphics_queue(physical_device, i)) {
                    accept = false;
                    break;
                }
            }
            if (accept)
                queue_idx_graphics = i;
        }
        if ((queue_idx_transfer < 0 || queue_idx_graphics == queue_idx_transfer) && supports_transfer) {
            queue_idx_transfer = i;
        }
    }

    if (queue_idx_graphics < 0 || queue_idx_transfer < 0) {
        throw std::runtime_error("could not find a suitable Vulkan queue family!");
    } else {
        spdlog::debug("found suitable vulkan queue families: {} {}", queue_idx_graphics, queue_idx_transfer);
        this->queue_idx_graphics = queue_idx_graphics;
        this->queue_idx_transfer = queue_idx_transfer;
    }
}

void Context::create_device_and_queues() {
    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    float queue_priority = 1.0f;

    queue_create_infos.push_back({{}, queue_idx_graphics, 1U, &queue_priority});
    if (queue_idx_graphics != queue_idx_transfer) {
        queue_create_infos.push_back({{}, queue_idx_transfer, 1U, &queue_priority});
    }

    std::vector<const char*> required_device_extensions;
    for (auto& ext : extensions) {
        insert_all(required_device_extensions, ext->required_device_extension_names());
    }
    spdlog::debug("required device extensions: [{}]", fmt::join(required_device_extensions, ", "));

    void* p_next = nullptr;
    for (auto& ext : extensions) {
        p_next = ext->on_create_device(p_next);
    }
    vk::PhysicalDeviceFeatures2 physical_device_features_2 = physical_device.getFeatures2();
    physical_device_features_2.pNext = p_next;

    vk::DeviceCreateInfo device_create_info{
        {}, queue_create_infos, layer_names, required_device_extensions, nullptr, &physical_device_features_2};

    device = physical_device.createDevice(device_create_info);
    spdlog::debug("device created");

    for (auto& ext : extensions) {
        ext->on_device_created(device);
    }

    queue_graphics = device.getQueue(queue_idx_graphics, 0);
    queue_transfer = device.getQueue(queue_idx_transfer, 0);
    spdlog::debug("queues created");
}

void Context::create_command_pools() {
    vk::CommandPoolCreateInfo cpi_graphics({vk::CommandPoolCreateFlagBits::eResetCommandBuffer}, queue_idx_graphics);
    cmd_pool_graphics = device.createCommandPool(cpi_graphics);
    vk::CommandPoolCreateInfo cpi_transfer({vk::CommandPoolCreateFlagBits::eResetCommandBuffer}, queue_idx_transfer);
    cmd_pool_transfer = device.createCommandPool(cpi_transfer);
    spdlog::debug("command pools created");
}
