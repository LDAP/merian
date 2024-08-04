#include "merian/vk/context.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/utils/pointer.hpp"

#include <spdlog/spdlog.h>
#include <tuple>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace merian {

ContextHandle Context::create(const std::vector<std::shared_ptr<Extension>>& extensions,
                                    const std::string& application_name,
                                    uint32_t application_vk_version,
                                    uint32_t preffered_number_compute_queues,
                                    uint32_t filter_vendor_id,
                                    uint32_t filter_device_id,
                                    const std::string& filter_device_name) {

    auto shared_context = std::shared_ptr<Context>(new Context(
        extensions, application_name, application_vk_version, preffered_number_compute_queues,
        filter_vendor_id, filter_device_id, filter_device_name));

    for (auto& ext : shared_context->extensions) {
        ext.second->on_context_created(shared_context);
    }

    return shared_context;
}

Context::Context(const std::vector<std::shared_ptr<Extension>>& desired_extensions,
                 const std::string& application_name,
                 uint32_t application_vk_version,
                 uint32_t preffered_number_compute_queues,
                 uint32_t filter_vendor_id,
                 uint32_t filter_device_id,
                 const std::string& filter_device_name)
    : application_name(application_name), application_vk_version(application_vk_version) {
    SPDLOG_INFO("\n\n\
__  __ ___ ___ ___   _   _  _ \n\
|  \\/  | __| _ \\_ _| /_\\ | \\| |\n\
| |\\/| | _||   /| | / _ \\| .` |\n\
|_|  |_|___|_|_\\___/_/ \\_\\_|\\_|\n\n\
Version: {}\n\n",
                MERIAN_VERSION);
    SPDLOG_INFO("context initializing...");

    // Init dynamic loader
    static vk::DynamicLoader dl;
    static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    SPDLOG_DEBUG("supplied extensions:");
    for (auto& ext : desired_extensions) {
        SPDLOG_DEBUG("{}", ext->name);
        if (extensions.contains(typeindex_from_pointer(ext))) {
            std::runtime_error{"A extension type can only be added once."};
        } else {
            extensions[typeindex_from_pointer(ext)] = ext;
        }
    }

    create_instance();
    prepare_physical_device(filter_vendor_id, filter_device_id, filter_device_name);
    find_queues();
    create_device_and_queues(preffered_number_compute_queues);

    SPDLOG_INFO("context ready.");
}

Context::~Context() {
    device.waitIdle();

    SPDLOG_DEBUG("destroy context");
    for (auto& ext : extensions) {
        ext.second->on_destroy_context();
    }

    SPDLOG_DEBUG("destroy pipeline cache");
    device.destroyPipelineCache(pipeline_cache);

    SPDLOG_DEBUG("destroy device");
    for (auto& ext : extensions) {
        ext.second->on_destroy_device(device);
    }
    device.destroy();

    SPDLOG_DEBUG("destroy instance");
    for (auto& ext : extensions) {
        ext.second->on_destroy_instance(instance);
    }
    instance.destroy();

    SPDLOG_INFO("context destroyed");
}

void Context::create_instance() {
    extensions_check_instance_layer_support();
    extensions_check_instance_extension_support();

    for (auto& ext : extensions) {
        insert_all(instance_layer_names, ext.second->required_instance_layer_names());
        insert_all(instance_extension_names, ext.second->required_instance_extension_names());
    }
    remove_duplicates(instance_layer_names);
    remove_duplicates(instance_extension_names);

    SPDLOG_DEBUG("enabling instance layers: [{}]", fmt::join(instance_layer_names, ", "));
    SPDLOG_DEBUG("enabling instance extensions: [{}]", fmt::join(instance_extension_names, ", "));

    void* p_next = nullptr;
    for (auto& ext : extensions) {
        p_next = ext.second->pnext_instance_create_info(p_next);
    }

    vk::ApplicationInfo application_info{
        application_name.c_str(),
        application_vk_version,
        MERIAN_PROJECT_NAME,
        VK_MAKE_VERSION(MERIAN_VERSION_MAJOR, MERIAN_VERSION_MINOR, MERIAN_VERSION_PATCH),
        vk_api_version,
    };

    vk::InstanceCreateInfo instance_create_info{
        {}, &application_info, instance_layer_names, instance_extension_names, p_next,
    };

    instance = vk::createInstance(instance_create_info);
    SPDLOG_DEBUG("instance created");

    // Must happen before on_instance_created since it requires dynamic loading
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
    for (auto& ext : extensions) {
        ext.second->on_instance_created(instance);
    }
}

void Context::prepare_physical_device(uint32_t filter_vendor_id,
                                      uint32_t filter_device_id,
                                      std::string filter_device_name) {
    const std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No vulkan device found!");
    }

    // Check environment variables
    if (const char* env_vendor_id = std::getenv("MERIAN_DEFAULT_FILTER_VENDOR_ID");
        filter_vendor_id == (uint32_t)-1 && env_vendor_id) {
        filter_vendor_id = std::strtoul(env_vendor_id, nullptr, 10);
    }
    if (const char* env_device_id = std::getenv("MERIAN_DEFAULT_FILTER_DEVICE_ID");
        filter_device_id == (uint32_t)-1 && env_device_id) {
        filter_device_id = std::strtoul(env_device_id, nullptr, 10);
    }
    if (const char* env_device_name = std::getenv("MERIAN_DEFAULT_FILTER_DEVICE_NAME");
        filter_device_name.empty() && env_device_name) {
        filter_device_name = env_device_name;
    }

    std::vector<std::tuple<vk::PhysicalDevice, vk::PhysicalDeviceProperties2>> matches;
    for (std::size_t i = 0; i < devices.size(); i++) {
        vk::PhysicalDeviceProperties2 props = devices[i].getProperties2();
        SPDLOG_INFO("found physical device {}, vendor id: {}, device id: {}",
                    props.properties.deviceName.data(), props.properties.vendorID,
                    props.properties.deviceID);
        if ((filter_vendor_id == (uint32_t)-1 || filter_vendor_id == props.properties.vendorID) &&
            (filter_device_id == (uint32_t)-1 || filter_device_id == props.properties.deviceID) &&
            (filter_device_name == "" || filter_device_name == props.properties.deviceName)) {
            matches.emplace_back(devices[i], props);
        }
    }

    if (matches.empty()) {
        throw std::runtime_error(fmt::format(
            "no vulkan device found with vendor id: {}, device id: {}, device name: {}.",
            filter_vendor_id == (uint32_t)-1 ? "any" : std::to_string(filter_vendor_id),
            filter_device_id == (uint32_t)-1 ? "any" : std::to_string(filter_device_id),
            filter_device_name.empty() ? "any" : filter_device_name));
    }
    std::sort(matches.begin(), matches.end(),
              [](std::tuple<vk::PhysicalDevice, vk::PhysicalDeviceProperties2>& a,
                 std::tuple<vk::PhysicalDevice, vk::PhysicalDeviceProperties2>& b) {
                  const vk::PhysicalDeviceProperties& props_a = std::get<1>(a).properties;
                  const vk::PhysicalDeviceProperties& props_b = std::get<1>(b).properties;
                  if (props_a.deviceType != props_b.deviceType) {
                      if (props_a.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
                          return false;
                      if (props_b.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
                          return true;
                      if (props_a.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
                          return false;
                      if (props_b.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
                          return true;
                  }
                  return std::get<0>(a).enumerateDeviceExtensionProperties().size() <
                         std::get<0>(b).enumerateDeviceExtensionProperties().size();

                  return false;
              });

    physical_device.physical_device = std::get<0>(matches.back());

    physical_device.physical_device_properties.pNext =
        &physical_device.physical_device_subgroup_properties;
    // ^
    physical_device.physical_device.getProperties2(&physical_device.physical_device_properties);
    SPDLOG_INFO("selected physical device {}, vendor id: {}, device id: {}",
                physical_device.physical_device_properties.properties.deviceName.data(),
                physical_device.physical_device_properties.properties.vendorID,
                physical_device.physical_device_properties.properties.deviceID);

    void* extension_features_pnext = nullptr;
    for (auto& ext : extensions) {
        extension_features_pnext = ext.second->pnext_get_features_2(extension_features_pnext);
    }
    // ^
    physical_device.features.physical_device_features_v13.setPNext(extension_features_pnext);
    // ^
    physical_device.features.physical_device_features_v12.setPNext(
        &physical_device.features.physical_device_features_v13);
    // ^
    physical_device.features.physical_device_features_v11.setPNext(
        &physical_device.features.physical_device_features_v12);
    // ^
    physical_device.features.physical_device_features.setPNext(
        &physical_device.features.physical_device_features_v11);
    // ^
    physical_device.physical_device.getFeatures2(
        &physical_device.features.physical_device_features);

    physical_device.physical_device_memory_properties =
        physical_device.physical_device.getMemoryProperties2();

    physical_device.physical_device_extension_properties =
        physical_device.physical_device.enumerateDeviceExtensionProperties();

    for (auto& ext : extensions) {
        ext.second->on_physical_device_selected(physical_device);
    }

    extensions_check_device_extension_support();
    extensions_self_check_support();
}

void Context::find_queues() {
    std::vector<vk::QueueFamilyProperties> queue_family_props =
        physical_device.physical_device.getQueueFamilyProperties();

    if (queue_family_props.empty()) {
        throw std::runtime_error{"no queue families available!"};
    }
    SPDLOG_DEBUG("number of queue families available: {}", queue_family_props.size());

    using Flags = vk::QueueFlagBits;

    // We calculate all possible index candidates then sort descending the list to get the best
    // match. (GCT found, additional T found, additional C found, number remaining compute queues,
    // GCT family index, T family index, C family index)
    std::vector<std::tuple<bool, bool, bool, uint32_t, uint32_t, uint32_t, uint32_t>> candidates;

#ifndef NDEBUG
    for (std::size_t i = 0; i < queue_family_props.size(); i++) {
        const bool supports_graphics =
            queue_family_props[i].queueFlags & Flags::eGraphics ? true : false;
        const bool supports_transfer =
            queue_family_props[i].queueFlags & Flags::eTransfer ? true : false;
        const bool supports_compute =
            queue_family_props[i].queueFlags & Flags::eCompute ? true : false;
        SPDLOG_DEBUG("queue family {}: supports graphics: {} transfer: {} compute: {}, count {}", i,
                     supports_graphics, supports_transfer, supports_compute,
                     queue_family_props[i].queueCount);
    }
#endif

    std::vector<uint32_t> queue_counts(queue_family_props.size());
    for (uint32_t i = 0; i < queue_family_props.size(); i++)
        queue_counts[i] = queue_family_props[i].queueCount;

    for (uint32_t queue_family_idx_GCT = 0; queue_family_idx_GCT < queue_family_props.size();
         queue_family_idx_GCT++) {
        for (uint32_t queue_family_idx_T = 0; queue_family_idx_T < queue_family_props.size();
             queue_family_idx_T++) {
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
                        return ext.second->accept_graphics_queue(
                            instance, physical_device.physical_device, queue_family_idx_GCT);
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
                    // we do not need to reduce remaining_queue_count[queue_family_idx_C] since its
                    // the last prio get number remaining instead
                    num_compute_queues = remaining_queue_count[queue_family_idx_C];
                }

                candidates.emplace_back(found_GCT, found_T, found_C, num_compute_queues,
                                        queue_family_idx_GCT, queue_family_idx_T,
                                        queue_family_idx_C);
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
        SPDLOG_WARN("not all requested queue families found! GCT: {} T: {} C: {}", found_GCT,
                    found_T, found_C);
    }

    this->queue_family_idx_GCT = found_GCT ? std::get<4>(best) : -1;
    this->queue_family_idx_T = found_T ? std::get<5>(best) : -1;
    this->queue_family_idx_C = found_C ? std::get<6>(best) : -1;

    SPDLOG_DEBUG("determined queue families indices: GCT: {} T: {} C: {}", queue_family_idx_GCT,
                 queue_family_idx_T, queue_family_idx_C);
}

void enable_common_features(const Context::FeaturesContainer& supported,
                            Context::FeaturesContainer& enable) {
    if (supported.physical_device_features_v11.storageBuffer16BitAccess) {
        SPDLOG_DEBUG("storageBuffer16BitAccess supported. Enabling feature");
        enable.physical_device_features_v11.storageBuffer16BitAccess = true;
    }

    if (supported.physical_device_features_v12.scalarBlockLayout) {
        SPDLOG_DEBUG("scalarBlockLayout supported. Enabling feature");
        enable.physical_device_features_v12.scalarBlockLayout = true;
    }
    if (supported.physical_device_features_v12.shaderFloat16) {
        SPDLOG_DEBUG("shaderFloat16 supported. Enabling feature");
        enable.physical_device_features_v12.shaderFloat16 = true;
    }
    if (supported.physical_device_features_v12.uniformAndStorageBuffer8BitAccess) {
        SPDLOG_DEBUG("uniformAndStorageBuffer8BitAccess supported. Enabling feature");
        enable.physical_device_features_v12.uniformAndStorageBuffer8BitAccess = true;
    }
    if (supported.physical_device_features_v12.bufferDeviceAddress) {
        SPDLOG_DEBUG("bufferDeviceAddress supported. Enabling feature");
        enable.physical_device_features_v12.bufferDeviceAddress = true;
    }

    if (supported.physical_device_features_v12.runtimeDescriptorArray) {
        SPDLOG_DEBUG("runtimeDescriptorArray supported. Enabling feature");
        enable.physical_device_features_v12.runtimeDescriptorArray = true;
    }
    if (supported.physical_device_features_v12.descriptorIndexing) {
        SPDLOG_DEBUG("descriptorIndexing supported. Enabling feature");
        enable.physical_device_features_v12.descriptorIndexing = true;
    }
    if (supported.physical_device_features_v12.shaderSampledImageArrayNonUniformIndexing) {
        SPDLOG_DEBUG("shaderSampledImageArrayNonUniformIndexing supported. Enabling feature");
        enable.physical_device_features_v12.shaderSampledImageArrayNonUniformIndexing = true;
    }
    if (supported.physical_device_features_v12.shaderStorageImageArrayNonUniformIndexing) {
        SPDLOG_DEBUG("shaderStorageImageArrayNonUniformIndexing supported. Enabling feature");
        enable.physical_device_features_v12.shaderStorageImageArrayNonUniformIndexing = true;
    }
    if (supported.physical_device_features_v12.shaderStorageBufferArrayNonUniformIndexing) {
        SPDLOG_DEBUG("shaderStorageBufferArrayNonUniformIndexing supported. Enabling feature");
        enable.physical_device_features_v12.shaderStorageBufferArrayNonUniformIndexing = true;
    }
    if (supported.physical_device_features_v12.shaderUniformBufferArrayNonUniformIndexing) {
        SPDLOG_DEBUG("shaderUniformBufferArrayNonUniformIndexing supported. Enabling feature");
        enable.physical_device_features_v12.shaderUniformBufferArrayNonUniformIndexing = true;
    }
    if (supported.physical_device_features_v12.shaderInt8) {
        SPDLOG_DEBUG("shaderInt8 supported. Enabling feature");
        enable.physical_device_features_v12.shaderInt8 = true;
    }
    if (supported.physical_device_features_v12.timelineSemaphore) {
        SPDLOG_DEBUG("timelineSemaphore supported. Enabling feature");
        enable.physical_device_features_v12.timelineSemaphore = true;
    }
    if (supported.physical_device_features_v12.hostQueryReset) {
        SPDLOG_DEBUG("hostQueryReset supported. Enabling feature");
        enable.physical_device_features_v12.hostQueryReset = true;
    }

    if (supported.physical_device_features_v13.robustImageAccess) {
        SPDLOG_DEBUG("robustImageAccess supported. Enabling feature");
        enable.physical_device_features_v13.robustImageAccess = true;
    }
    if (supported.physical_device_features_v13.synchronization2) {
        SPDLOG_DEBUG("synchronization2 supported. Enabling feature");
        enable.physical_device_features_v13.synchronization2 = true;
    }
    if (supported.physical_device_features_v13.maintenance4) {
        SPDLOG_DEBUG("maintenance4 supported. Enabling feature");
        enable.physical_device_features_v13.maintenance4 = true;
    }
}

void Context::create_device_and_queues(uint32_t preferred_number_compute_queues) {
    // PREPARE QUEUES

    std::vector<vk::QueueFamilyProperties> queue_family_props =
        physical_device.physical_device.getQueueFamilyProperties();
    std::vector<uint32_t> count_per_family(queue_family_props.size());
    uint32_t actual_number_compute_queues = 0;

    if (queue_family_idx_GCT >= 0) {
        queue_idx_GCT = count_per_family[queue_family_idx_GCT]++;
        SPDLOG_DEBUG("queue index GCT: {}", queue_idx_GCT);
    }
    if (queue_family_idx_T >= 0) {
        queue_idx_T = count_per_family[queue_family_idx_T]++;
        SPDLOG_DEBUG("queue index T: {}", queue_idx_T);
    }
    if (queue_family_idx_C >= 0) {
        uint32_t remaining_compute_queues = queue_family_props[queue_family_idx_C].queueCount -
                                            count_per_family[queue_family_idx_C];
        actual_number_compute_queues =
            std::min(remaining_compute_queues, preferred_number_compute_queues);

        for (uint32_t i = 0; i < actual_number_compute_queues; i++) {
            queue_idx_C.emplace_back(count_per_family[queue_family_idx_C]++);
        }
        SPDLOG_DEBUG("queue indices C: [{}]", fmt::join(queue_idx_C, ", "));
    }
    queues_C.resize(actual_number_compute_queues);

    uint32_t max_queue_count = *std::max_element(count_per_family.begin(), count_per_family.end());
    std::vector<float> queue_priorities(max_queue_count, 1.0f);

    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    for (uint32_t queue_familiy_idx = 0; queue_familiy_idx < queue_family_props.size();
         queue_familiy_idx++) {
        if (count_per_family[queue_familiy_idx] > 0) {
            queue_create_infos.push_back({{},
                                          queue_familiy_idx,
                                          count_per_family[queue_familiy_idx],
                                          queue_priorities.data()});
        }
    }

    // DEVICE EXTENSIONS
    for (auto& ext : extensions) {
        insert_all(device_extensions,
                   ext.second->required_device_extension_names(physical_device.physical_device));
    }
    remove_duplicates(device_extensions);
    SPDLOG_DEBUG("enabling device extensions: [{}]", fmt::join(device_extensions, ", "));

    // FEATURES

    FeaturesContainer enable;
    // TODO: This enables all features which may be overkill
    enable.physical_device_features = physical_device.features.physical_device_features;
    enable_common_features(physical_device.features, enable);
    for (auto& ext : extensions) {
        ext.second->enable_device_features(physical_device.features, enable);
    }

    // Setup p_next for extensions

    // Extensions can enable features of their extensions
    void* extensions_device_create_p_next = nullptr;
    for (auto& ext : extensions) {
        extensions_device_create_p_next =
            ext.second->pnext_device_create_info(extensions_device_create_p_next);
    }
    // ^
    enable.physical_device_features_v13.setPNext(extensions_device_create_p_next);
    // ^
    enable.physical_device_features_v12.setPNext(&enable.physical_device_features_v13);
    // ^
    enable.physical_device_features_v11.setPNext(&enable.physical_device_features_v12);
    // ^
    enable.physical_device_features.setPNext(&enable.physical_device_features_v11);
    // ^
    vk::DeviceCreateInfo device_create_info{{},
                                            queue_create_infos,
                                            instance_layer_names,
                                            device_extensions,
                                            nullptr,
                                            &enable.physical_device_features};

    device = physical_device.physical_device.createDevice(device_create_info);
    SPDLOG_DEBUG("device created and queues created");

    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
    for (auto& ext : extensions) {
        ext.second->on_device_created(device);
    }

    SPDLOG_DEBUG("create pipeline cache");
    vk::PipelineCacheCreateInfo pipeline_cache_create_info{};
    pipeline_cache = device.createPipelineCache(pipeline_cache_create_info);
}

////////////
// HELPERS
////////////

void Context::extensions_check_instance_layer_support() {
    SPDLOG_DEBUG("extensions: checking instance layer support...");
    std::vector<std::shared_ptr<Extension>> not_supported;
    std::vector<vk::LayerProperties> layer_props = vk::enumerateInstanceLayerProperties();

    for (auto& ext : extensions) {
        std::vector<const char*> layers = ext.second->required_instance_layer_names();
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
            not_supported.push_back(ext.second);
            ext.second->on_unsupported("instance layer missing");
        }
    }
    destroy_extensions(not_supported);
}

void Context::extensions_check_instance_extension_support() {
    SPDLOG_DEBUG("extensions: checking instance extension support...");
    std::vector<std::shared_ptr<Extension>> not_supported;
    std::vector<vk::ExtensionProperties> extension_props =
        vk::enumerateInstanceExtensionProperties();

    for (auto& ext : extensions) {
        std::vector<const char*> instance_extensions =
            ext.second->required_instance_extension_names();
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
            not_supported.push_back(ext.second);
            ext.second->on_unsupported("instance extension missing");
        }
    }
    destroy_extensions(not_supported);
}

void Context::extensions_check_device_extension_support() {
    SPDLOG_DEBUG("extensions: checking device extension support...");
    std::vector<std::shared_ptr<Extension>> not_supported;

    for (auto& ext : extensions) {
        std::vector<const char*> device_extensions =
            ext.second->required_device_extension_names(physical_device.physical_device);
        bool all_extensions_found = true;
        for (auto& layer : device_extensions) {
            bool extension_found = false;
            for (auto& extension_prop : physical_device.physical_device_extension_properties) {
                if (!strcmp(extension_prop.extensionName, layer)) {
                    extension_found = true;
                    break;
                }
            }
            all_extensions_found &= extension_found;
        }
        if (!all_extensions_found) {
            not_supported.push_back(ext.second);
            ext.second->on_unsupported("device extension missing");
        }
    }
    destroy_extensions(not_supported);
}

void Context::extensions_self_check_support() {
    SPDLOG_DEBUG("extensions: self-check support...");
    std::vector<std::shared_ptr<Extension>> not_supported;
    for (auto& ext : extensions) {
        if (!ext.second->extension_supported(physical_device)) {
            ext.second->on_unsupported("self-check failed");
            not_supported.push_back(ext.second);
        }
    }
    destroy_extensions(not_supported);
}

void Context::destroy_extensions(std::vector<std::shared_ptr<Extension>> extensions) {
    for (auto& ext : extensions) {
        SPDLOG_DEBUG("remove extension {} from context", ext->name);
        auto& r = *ext.get(); // suppress warnings...
        this->extensions.erase(typeid(r));
    }
}

///////////////
// GETTER
///////////////

uint32_t Context::get_number_compute_queues() const noexcept {
    return queues_C.size();
}

std::shared_ptr<Queue> Context::get_queue_GCT() {
    if (!queue_GCT.expired()) {
        return queue_GCT.lock();
    } else if (queue_family_idx_GCT < 0) {
        return nullptr;
    } else {
        auto queue =
            std::make_shared<Queue>(shared_from_this(), queue_family_idx_GCT, queue_idx_GCT);
        queue_GCT = queue;
        return queue;
    }
}

std::shared_ptr<Queue> Context::get_queue_T(const bool fallback) {
    if (queue_family_idx_T < 0) {
        if (fallback)
            return get_queue_GCT();
        else
            return nullptr;
    }
    if (!queue_T.expired()) {
        return queue_T.lock();
    } else {
        auto queue = std::make_shared<Queue>(shared_from_this(), queue_family_idx_T, queue_idx_T);
        queue_T = queue;
        return queue;
    }
}

std::shared_ptr<Queue> Context::get_queue_C(uint32_t index, const bool fallback) {
    assert(fallback || index < queues_C.size());

    if (index < queues_C.size()) {
        if (!queues_C[index].expired()) {
            return queues_C[index].lock();
        } else {
            auto queue =
                std::make_shared<Queue>(shared_from_this(), queue_family_idx_C, queue_idx_C[index]);
            queues_C[index] = queue;
            return queue;
        }
    } else if (!fallback) {
        // early out, fallback is not allowed
        return nullptr;
    } else if (!queues_C.empty()) {
        auto unused_queue = std::find_if(queues_C.begin(), queues_C.end(),
                                         [](auto& queue) { return queue.expired(); });
        if (unused_queue == queues_C.end()) {
            // there is no unused queue, use first
            return get_queue_C(0);
        } else {
            return get_queue_C(unused_queue - queues_C.begin());
        }
    } else {
        // there are no extra compute queues, maybe at least a graphics queue with compute support
        return get_queue_GCT();
    }
}

std::shared_ptr<CommandPool> Context::get_cmd_pool_GCT() {
    if (!cmd_pool_GCT.expired()) {
        return cmd_pool_GCT.lock();
    } else {
        auto cmd = std::make_shared<CommandPool>(get_queue_GCT());
        cmd_pool_GCT = cmd;
        return cmd;
    }
}

std::shared_ptr<CommandPool> Context::get_cmd_pool_T() {
    if (!cmd_pool_T.expired()) {
        return cmd_pool_T.lock();
    } else {
        auto cmd = std::make_shared<CommandPool>(get_queue_T());
        cmd_pool_T = cmd;
        return cmd;
    }
}

std::shared_ptr<CommandPool> Context::get_cmd_pool_C() {
    if (!cmd_pool_C.expired()) {
        return cmd_pool_C.lock();
    } else {
        auto cmd = std::make_shared<CommandPool>(get_queue_C());
        cmd_pool_C = cmd;
        return cmd;
    }
}

bool Context::device_extension_enabled(const std::string& name) {
    return std::find_if(device_extensions.begin(), device_extensions.end(),
                        [&](const char* s) { return name == s; }) != device_extensions.end();
}

bool Context::instance_extension_enabled(const std::string& name) {
    return std::find_if(instance_extension_names.begin(), instance_extension_names.end(),
                        [&](const char* s) { return name == s; }) != instance_extension_names.end();
}

} // namespace merian
