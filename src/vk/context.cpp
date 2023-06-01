#include "merian/vk/context.hpp"
#include "merian/utils/vector_utils.hpp"
#include "merian/vk/extension/extension.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <tuple>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace merian {

Context::Context(std::vector<Extension*> desired_extensions,
                 std::string application_name,
                 uint32_t application_vk_version,
                 uint32_t preffered_number_compute_queues,
                 uint32_t filter_vendor_id,
                 uint32_t filter_device_id,
                 std::string filter_device_name)
    : extensions(desired_extensions) {
    assert(preffered_number_compute_queues >= 0);

    SPDLOG_INFO("This is {} {}. Context initializing...", MERIAN_PROJECT_NAME, MERIAN_VERSION);

    // Init dynamic loader
    static vk::DynamicLoader dl;
    static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    SPDLOG_TRACE("supplied extensions:");
    for (auto& ext : extensions) {
        (void)ext;
        SPDLOG_TRACE("{}", ext->name);
    }

    create_instance(application_name, application_vk_version);
    prepare_physical_device(filter_vendor_id, filter_device_id, filter_device_name);
    find_queues();
    create_device_and_queues(preffered_number_compute_queues);
    create_command_pools();

    for (auto& ext : extensions) {
        ext->on_context_created(*this);
    }

    SPDLOG_INFO("Context ready.");
}

Context::~Context() {
    device.waitIdle();

    SPDLOG_DEBUG("destroy context");
    for (auto& ext : extensions) {
        ext->on_destroy_context(*this);
    }

    SPDLOG_DEBUG("destroy command pools");
    if (cmd_pool_GCT.has_value())
        device.destroyCommandPool(cmd_pool_GCT.value());
    if (cmd_pool_T.has_value())
        device.destroyCommandPool(cmd_pool_T.value());
    if (cmd_pool_C.has_value())
        device.destroyCommandPool(cmd_pool_C.value());

    SPDLOG_DEBUG("destroy queues");
    if (!queue_GCT.unique())
        SPDLOG_WARN("graphics queue shared-ptr is not unique. Make sure to release your references!");
    queue_GCT.reset();
    if (!queue_T.unique())
        SPDLOG_WARN("transfer queue shared-ptr is not unique. Make sure to release your references!");
    queue_T.reset();

    queue_C.reset();
    for (std::size_t i = 0; i < queues_C.size(); i++) {
        if (!queues_C[i].unique())
            SPDLOG_WARN("compute queue {} shared-ptr is not unique. Make sure to release your references!", i);
    }
    queues_C.clear();

    SPDLOG_DEBUG("destroy device");
    for (auto& ext : extensions) {
        ext->on_destroy_device(device);
    }
    device.destroy();

    SPDLOG_DEBUG("destroy instance");
    for (auto& ext : extensions) {
        ext->on_destroy_instance(instance);
    }
    instance.destroy();

    SPDLOG_DEBUG("context destroyed");
}

void Context::create_instance(std::string application_name, uint32_t application_vk_version) {
    extensions_check_instance_layer_support();
    extensions_check_instance_extension_support();

    for (auto& ext : extensions) {
        insert_all(instance_layer_names, ext->required_instance_layer_names());
        insert_all(instance_extension_names, ext->required_instance_extension_names());
    }
    remove_duplicates(instance_layer_names);
    remove_duplicates(instance_extension_names);

    SPDLOG_DEBUG("enabling instance layers: [{}]", fmt::join(instance_layer_names, ", "));
    SPDLOG_DEBUG("enabling instance extensions: [{}]", fmt::join(instance_extension_names, ", "));

    void* p_next = nullptr;
    for (auto& ext : extensions) {
        p_next = ext->on_create_instance(p_next);
    }

    vk::ApplicationInfo application_info{
        application_name.c_str(), application_vk_version,
        MERIAN_PROJECT_NAME,      VK_MAKE_VERSION(MERIAN_VERSION_MAJOR, MERIAN_VERSION_MINOR, MERIAN_VERSION_PATCH),
        VK_API_VERSION_1_3,
    };

    vk::InstanceCreateInfo instance_create_info{
        {},
        &application_info,
        static_cast<uint32_t>(instance_layer_names.size()),
        instance_layer_names.data(),
        static_cast<uint32_t>(instance_extension_names.size()),
        instance_extension_names.data(),
        p_next,
    };

    instance = vk::createInstance(instance_create_info);
    SPDLOG_DEBUG("instance created");

    // Must happen before on_instance_created since it requires dynamic loading
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
    for (auto& ext : extensions) {
        ext->on_instance_created(instance);
    }
}

void Context::prepare_physical_device(uint32_t filter_vendor_id,
                                      uint32_t filter_device_id,
                                      std::string filter_device_name) {
    std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No vulkan device found!");
    }

    // TODO: Sort by score (e.g. prefer dedicated GPU, check supported extensions)
    long selected = -1;
    for (std::size_t i = 0; i < devices.size(); i++) {
        vk::PhysicalDeviceProperties2 props = devices[i].getProperties2();
        SPDLOG_INFO("found physical device {}, vendor id: {}, device id: {}", props.properties.deviceName,
                    props.properties.vendorID, props.properties.deviceID);
        if ((filter_vendor_id == (uint32_t)-1 || filter_vendor_id == props.properties.vendorID) &&
            (filter_device_id == (uint32_t)-1 || filter_device_id == props.properties.deviceID) &&
            (filter_device_name == "" || filter_device_name == props.properties.deviceName)) {
            selected = i;
        }
    }

    if (selected == -1) {
        throw std::runtime_error(fmt::format("no vulkan device found with vendor id: {}, device id: {}! (-1 means any)",
                                             filter_vendor_id, filter_device_id));
    }
    physical_device = devices[selected];
    physical_device_props = physical_device.getProperties();
    SPDLOG_INFO("selected physical device {}, vendor id: {}, device id: {}",
                physical_device_props.properties.deviceName, physical_device_props.properties.vendorID,
                physical_device_props.properties.deviceID);

    physical_device_features = physical_device.getFeatures2();
    physical_device_memory_properties = physical_device.getMemoryProperties2();
    physical_device_extension_properties = physical_device.enumerateDeviceExtensionProperties();

    for (auto& ext : extensions) {
        ext->on_physical_device_selected(physical_device);
    }

    extensions_check_device_extension_support();
    extensions_self_check_support();
}

void Context::find_queues() {
    std::vector<vk::QueueFamilyProperties> queue_family_props = physical_device.getQueueFamilyProperties();

    if (queue_family_props.empty()) {
        throw std::runtime_error{"no queue families available!"};
    }
    SPDLOG_DEBUG("number of queue families available: {}", queue_family_props.size());

    using Flags = vk::QueueFlagBits;

    // We calculate all possible index candidates then sort descending the list to get the best match.
    // (GCT found, additional T found, additional C found, number remaining compute queues, GCT family index, T family
    // index, C family index)
    std::vector<std::tuple<bool, bool, bool, uint32_t, uint32_t, uint32_t, uint32_t>> candidates;

#ifdef DEBUG
    for (std::size_t i = 0; i < queue_family_props.size(); i++) {
        const bool supports_graphics = queue_family_props[i].queueFlags & Flags::eGraphics ? true : false;
        const bool supports_transfer = queue_family_props[i].queueFlags & Flags::eTransfer ? true : false;
        const bool supports_compute = queue_family_props[i].queueFlags & Flags::eCompute ? true : false;
        SPDLOG_DEBUG("queue family {}: supports graphics: {} transfer: {} compute: {}, count {}", i, supports_graphics,
                     supports_transfer, supports_compute, queue_family_props[i].queueCount);
    }
#endif

    std::vector<uint32_t> queue_counts(queue_family_props.size());
    for (uint32_t i = 0; i < queue_family_props.size(); i++)
        queue_counts[i] = queue_family_props[i].queueCount;

    for (uint32_t queue_family_idx_GCT = 0; queue_family_idx_GCT < queue_family_props.size(); queue_family_idx_GCT++) {
        for (uint32_t queue_family_idx_T = 0; queue_family_idx_T < queue_family_props.size(); queue_family_idx_T++) {
            for (uint32_t queue_family_idx_C = 0; queue_family_idx_C < queue_family_props.size();
                 queue_family_idx_C++) {

                // Make sure we do not request more queues that are available!
                std::vector<uint32_t> remaining_queue_count = queue_counts;
                bool found_GCT = false;
                bool found_T = false;
                bool found_C = false;
                uint32_t num_compute_queues = 0;

                // Prio 1: GCT
                if ((queue_family_props[queue_family_idx_GCT].queueFlags & Flags::eGraphics) &&
                    (queue_family_props[queue_family_idx_GCT].queueFlags & Flags::eCompute) &&
                    (queue_family_props[queue_family_idx_GCT].queueFlags & Flags::eTransfer) &&
                    remaining_queue_count[queue_family_idx_GCT] > 0 &&
                    std::all_of(extensions.begin(), extensions.end(), [&](auto ext) {
                        return ext->accept_graphics_queue(physical_device, queue_family_idx_GCT);
                    })) {
                    found_GCT = true;
                    remaining_queue_count[queue_family_idx_GCT]--;
                }
                // Prio 2: T (additional)
                if ((queue_family_props[queue_family_idx_T].queueFlags & Flags::eTransfer) &&
                    remaining_queue_count[queue_family_idx_T] > 0) {
                    found_T = true;
                    remaining_queue_count[queue_family_idx_T]--;
                }
                // Prio 3: C (additional)
                if ((queue_family_props[queue_family_idx_C].queueFlags & Flags::eCompute) &&
                    remaining_queue_count[queue_family_idx_C] > 0) {
                    found_C = true;
                    // we do not need to reduce remaining_queue_count[queue_family_idx_C] since its the last prio
                    // get number remaining instead
                    num_compute_queues = remaining_queue_count[queue_family_idx_C];
                }

                candidates.emplace_back(found_GCT, found_T, found_C, num_compute_queues, queue_family_idx_GCT,
                                        queue_family_idx_T, queue_family_idx_C);
            }
        }
    }

    // Descending order
    std::sort(candidates.begin(), candidates.end(), std::greater<>());
    auto best = candidates[0];

    bool found_GCT = std::get<0>(best);
    bool found_T = std::get<1>(best);
    bool found_C = std::get<2>(best);

    if (!found_GCT || !found_T || !found_C) {
        SPDLOG_WARN("not all requested queue families found! GCT: {} T: {} C: {}", found_GCT, found_T, found_C);
    }

    this->queue_family_idx_GCT = found_GCT ? std::get<4>(best) : -1;
    this->queue_family_idx_T = found_T ? std::get<5>(best) : -1;
    this->queue_family_idx_C = found_C ? std::get<6>(best) : -1;

    SPDLOG_DEBUG("determined queue families indices: GCT: {} T: {} C: {}", queue_family_idx_GCT, queue_family_idx_T,
                 queue_family_idx_C);
}

void Context::create_device_and_queues(uint32_t preferred_number_compute_queues) {
    std::vector<vk::QueueFamilyProperties> queue_family_props = physical_device.getQueueFamilyProperties();
    std::vector<uint32_t> count_per_family(queue_family_props.size());
    uint32_t actual_number_compute_queues = 0;
    uint32_t queue_idx_GCT = 0;
    uint32_t queue_idx_T = 0;
    std::vector<uint32_t> queue_idx_C;

    if (queue_family_idx_GCT >= 0) {
        queue_idx_GCT = count_per_family[queue_family_idx_GCT]++;
        SPDLOG_DEBUG("queue index GCT: {}", queue_idx_GCT);
    }
    if (queue_family_idx_T >= 0) {
        queue_idx_T = count_per_family[queue_family_idx_T]++;
        SPDLOG_DEBUG("queue index T: {}", queue_idx_T);
    }
    if (queue_family_idx_C >= 0) {
        uint32_t remaining_compute_queues =
            queue_family_props[queue_family_idx_C].queueCount - count_per_family[queue_family_idx_C];
        actual_number_compute_queues = std::min(remaining_compute_queues, preferred_number_compute_queues);

        for (uint32_t i = 0; i < actual_number_compute_queues; i++) {
            queue_idx_C.emplace_back(count_per_family[queue_family_idx_C]++);
        }
        SPDLOG_DEBUG("queue indices C: [{}]", fmt::join(queue_idx_C, ", "));
    }

    float queue_priority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    for (uint32_t queue_familiy_idx = 0; queue_familiy_idx < queue_family_props.size(); queue_familiy_idx++) {
        if (count_per_family[queue_familiy_idx] > 0) {
            queue_create_infos.push_back({{}, queue_familiy_idx, count_per_family[queue_familiy_idx], &queue_priority});
        }
    }

    std::vector<const char*> required_device_extensions;
    for (auto& ext : extensions) {
        insert_all(required_device_extensions, ext->required_device_extension_names());
    }
    remove_duplicates(required_device_extensions);
    SPDLOG_DEBUG("enabling device extensions: [{}]", fmt::join(required_device_extensions, ", "));

    void* p_next = nullptr;
    for (auto& ext : extensions) {
        p_next = ext->on_create_device(p_next);
    }
    // TODO: This enables all features which may be overkill
    vk::PhysicalDeviceFeatures2 physical_device_features_2 = physical_device.getFeatures2();
    physical_device_features_2.pNext = p_next;

    vk::DeviceCreateInfo device_create_info{
        {}, queue_create_infos, instance_layer_names, required_device_extensions, nullptr, &physical_device_features_2};

    device = physical_device.createDevice(device_create_info);
    SPDLOG_DEBUG("device created");

    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
    for (auto& ext : extensions) {
        ext->on_device_created(device);
    }

    if (queue_family_idx_GCT >= 0)
        queue_GCT = std::make_shared<QueueContainer>(device, queue_family_idx_GCT, queue_idx_GCT);
    if (queue_family_idx_T >= 0)
        queue_T = std::make_shared<QueueContainer>(device, queue_family_idx_T, queue_idx_T);
    if (queue_family_idx_T >= 0) {
        for (auto queue_idx : queue_idx_C) {
            queues_C.push_back(std::make_shared<QueueContainer>(device, queue_family_idx_C, queue_idx));
        }
        queue_C = queues_C[0];
    }

    SPDLOG_DEBUG("queues created");
}

void Context::create_command_pools() {
    if (queue_family_idx_GCT) {
        vk::CommandPoolCreateInfo cpi_graphics({vk::CommandPoolCreateFlagBits::eResetCommandBuffer},
                                               queue_family_idx_GCT);
        cmd_pool_GCT = device.createCommandPool(cpi_graphics);
    }
    if (queue_family_idx_T) {
        vk::CommandPoolCreateInfo cpi_transfer({vk::CommandPoolCreateFlagBits::eResetCommandBuffer},
                                               queue_family_idx_T);
        cmd_pool_T = device.createCommandPool(cpi_transfer);
    }
    if (queue_family_idx_C) {
        vk::CommandPoolCreateInfo cpi_compute({vk::CommandPoolCreateFlagBits::eResetCommandBuffer}, queue_family_idx_C);
        cmd_pool_C = device.createCommandPool(cpi_compute);
    }
    SPDLOG_DEBUG("command pools created");
}

////////////
// HELPERS
////////////

void Context::extensions_check_instance_layer_support() {
    SPDLOG_DEBUG("extensions: checking instance layer support...");
    std::vector<Extension*> not_supported;
    std::vector<vk::LayerProperties> layer_props = vk::enumerateInstanceLayerProperties();

    for (auto& ext : extensions) {
        std::vector<const char*> layers = ext->required_instance_layer_names();
        bool all_layers_found = true;
        for (auto& layer : layers) {
            bool layer_found = false;
            for (auto& layer_prop : layer_props) {
                if (!strcmp(layer_prop.layerName, layer)) {
                    layer_found = true;
                    break;
                }
            }
            all_layers_found &= layer_found;
        }
        if (!all_layers_found) {
            spdlog::warn("extension {} not supported (instance layer missing), disabling...", ext->name);
            not_supported.push_back(ext);
            ext->supported = false;
        }
    }
    destroy_extensions(not_supported);
}

void Context::extensions_check_instance_extension_support() {
    SPDLOG_DEBUG("extensions: checking instance extension support...");
    std::vector<Extension*> not_supported;
    std::vector<vk::ExtensionProperties> extension_props = vk::enumerateInstanceExtensionProperties();

    for (auto& ext : extensions) {
        std::vector<const char*> instance_extensions = ext->required_instance_extension_names();
        bool all_extensions_found = true;
        for (auto& layer : instance_extensions) {
            bool extension_found = false;
            for (auto& extension_prop : extension_props) {
                if (!strcmp(extension_prop.extensionName, layer)) {
                    extension_found = true;
                    break;
                }
            }
            all_extensions_found &= extension_found;
        }
        if (!all_extensions_found) {
            spdlog::warn("extension {} not supported (instance extension missing), disabling...", ext->name);
            not_supported.push_back(ext);
            ext->supported = false;
        }
    }
    destroy_extensions(not_supported);
}

void Context::extensions_check_device_extension_support() {
    SPDLOG_DEBUG("extensions: checking device extension support...");
    std::vector<Extension*> not_supported;
    std::vector<vk::ExtensionProperties> extension_props = physical_device.enumerateDeviceExtensionProperties();

    for (auto& ext : extensions) {
        std::vector<const char*> device_extensions = ext->required_device_extension_names();
        bool all_extensions_found = true;
        for (auto& layer : device_extensions) {
            bool extension_found = false;
            for (auto& extension_prop : extension_props) {
                if (!strcmp(extension_prop.extensionName, layer)) {
                    extension_found = true;
                    break;
                }
            }
            all_extensions_found &= extension_found;
        }
        if (!all_extensions_found) {
            spdlog::warn("extension {} not supported (device extension missing), disabling...", ext->name);
            not_supported.push_back(ext);
            ext->supported = false;
        }
    }
    destroy_extensions(not_supported);
}

void Context::extensions_self_check_support() {
    SPDLOG_DEBUG("extensions: self-check support...");
    std::vector<Extension*> not_supported;
    for (auto& ext : extensions) {
        if (!ext->extension_supported(physical_device)) {
            spdlog::warn("extension {} not supported (self-check failed), disabling...", ext->name);
            ext->supported = false;
            not_supported.push_back(ext);
        }
    }
    destroy_extensions(not_supported);
}

void Context::destroy_extensions(std::vector<Extension*> extensions) {
    for (auto& ext : extensions) {
        SPDLOG_DEBUG("remove extension {} from context", ext->name);
        ext->on_destroy_instance(this->instance);

        for (std::size_t i = 0; this->extensions.size(); i++) {
            if (this->extensions[i] == ext) {
                std::swap(this->extensions[i], this->extensions[this->extensions.size() - 1]);
                this->extensions.pop_back();
                break;
            }
        }
    }
}

} // namespace merian
