#include "merian/vk/context.hpp"
#include "merian/utils/pointer.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/extension_registry.hpp"
#include "merian/vk/utils/vulkan_extensions.hpp"
#include "merian/vk/utils/vulkan_spirv.hpp"

#include <fmt/ranges.h>
#include <numeric>
#include <queue>
#include <spdlog/spdlog.h>
#include <tuple>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace merian {

struct Context::DeviceSupportCache {
    std::unordered_map<std::shared_ptr<ContextExtension>, DeviceSupportInfo> results;
};

ContextHandle Context::create(const ContextCreateInfo& create_info) {
    const ContextHandle context = std::shared_ptr<Context>(new Context(create_info));

    for (const auto& ext : context->get_extensions()) {
        ext->on_context_created(context, *context);
    }

    return context;
}

void ExtensionContainer::add_extension(const std::shared_ptr<ContextExtension>& extension) {
    const std::type_index type_idx = typeindex_from_pointer(extension);
    context_extensions[type_idx] = extension;
    ordered_extensions.push_back(extension);
}

void ExtensionContainer::remove_extension(const std::type_index& type) {
    // Remove from map
    context_extensions.erase(type);

    auto it =
        std::remove_if(ordered_extensions.begin(), ordered_extensions.end(),
                       [&type](const auto& ext) { return std::type_index(typeid(*ext)) == type; });
    ordered_extensions.erase(it, ordered_extensions.end());

    assert(ordered_extensions.size() == context_extensions.size());
}

void Context::load_extensions(const std::vector<std::string>& extension_names) {
    std::unordered_set<std::string> loaded_extensions;
    std::unordered_map<std::string, std::shared_ptr<ContextExtension>> pending_extensions;
    std::queue<std::string> to_process;

    // Queue initial extensions
    to_process.push("merian");
    for (const auto& ext_name : extension_names) {
        to_process.push(ext_name);
    }

    // Process extensions, ensuring dependencies are loaded first
    while (!to_process.empty()) {
        const std::string ext_name = to_process.front();
        to_process.pop();

        // Skip if already loaded
        if (loaded_extensions.contains(ext_name)) {
            continue;
        }

        // Get or create extension
        std::shared_ptr<ContextExtension> ext;
        if (pending_extensions.contains(ext_name)) {
            ext = pending_extensions[ext_name];
        } else {
            ext = ExtensionRegistry::get_instance().create(ext_name);
            if (!ext) {
                throw MerianException{
                    fmt::format("Extension '{}' not found in registry", ext_name)};
            }
            pending_extensions[ext_name] = ext;
        }

        // Check if all dependencies are loaded
        auto requested = ext->request_extensions();
        bool all_deps_loaded = true;
        for (const auto& dep_name : requested) {
            if (!loaded_extensions.contains(dep_name)) {
                all_deps_loaded = false;
                // Queue dependency if not already queued or loaded
                if (!pending_extensions.contains(dep_name)) {
                    to_process.push(dep_name);
                }
            }
        }

        // If dependencies not ready, re-queue this extension for later
        if (!all_deps_loaded) {
            to_process.push(ext_name);
            continue;
        }

        // All dependencies loaded, add this extension
        SPDLOG_DEBUG("Loading extension: {}", ext_name);
        loaded_extensions.insert(ext_name);
        pending_extensions.erase(ext_name);
        add_extension(ext);
    }
}

Context::Context(const ContextCreateInfo& create_info)
    : application_name(create_info.application_name),
      application_vk_version(create_info.application_vk_version) {
    merian::Stopwatch sw;

    SPDLOG_INFO("\n\n\
__  __ ___ ___ ___   _   _  _ \n\
|  \\/  | __| _ \\_ _| /_\\ | \\| |\n\
| |\\/| | _||   /| | / _ \\| .` |\n\
|_|  |_|___|_|_\\___/_/ \\_\\_|\\_|\n\n\
Version: {}\n\n",
                MERIAN_VERSION);
    SPDLOG_INFO("context initializing...");

    SPDLOG_DEBUG("compiled with Vulkan header: {}.{}.{}",
                 VK_API_VERSION_MAJOR(VK_HEADER_VERSION_COMPLETE),
                 VK_API_VERSION_MINOR(VK_HEADER_VERSION_COMPLETE),
                 VK_API_VERSION_PATCH(VK_HEADER_VERSION_COMPLETE));

    SPDLOG_DEBUG("initializing dynamic loader");
    VULKAN_HPP_DEFAULT_DISPATCHER.init();

    load_extensions(create_info.context_extensions);

    // Invoke configuration callback if provided
    if (create_info.configure_extensions_callback) {
        (*create_info.configure_extensions_callback)(*this);
    }

    // Initialize file loader early so extensions can use it
    prepare_file_loader(create_info);

    for (const auto& ext : get_extensions()) {
        ext->on_context_initializing(VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
                                     file_loader, create_info);
    }

    const uint32_t target_vk_api_version = VK_HEADER_VERSION_COMPLETE;
    create_instance(target_vk_api_version, create_info.desired_features,
                    create_info.additional_extensions);

    auto support_cache =
        select_physical_device(create_info.filter_vendor_id, create_info.filter_device_id,
                               create_info.filter_device_name, create_info.desired_features,
                               create_info.additional_extensions);

    for (const auto& ext : get_extensions()) {
        ext->on_extension_support_confirmed(*this);
    }

    create_device_and_queues(create_info.preferred_number_compute_queues,
                             create_info.desired_features, create_info.additional_extensions,
                             support_cache);

    shader_compile_context = ShaderCompileContext::create(file_loader->get_search_paths(), device);

    SPDLOG_INFO("context ready. (took: {})", format_duration(sw.nanos()));
}

Context::~Context() {
    SPDLOG_INFO("context destroyed");
}

void Context::create_instance(const uint32_t targeted_vk_api_version,
                              const VulkanFeatures& desired_features,
                              const std::vector<const char*>& desired_additional_extensions) {
    const uint32_t effective_vk_instance_api_version =
        std::min(targeted_vk_api_version, Instance::get_instance_vk_api_version());

    std::unordered_set<std::string> supported_instance_layers;
    std::unordered_set<std::string> supported_instance_extensions;
    for (const auto& instance_layer : vk::enumerateInstanceLayerProperties()) {
        supported_instance_layers.emplace(instance_layer.layerName.data());
    }
    for (const auto& instance_ext : vk::enumerateInstanceExtensionProperties()) {
        supported_instance_extensions.emplace(instance_ext.extensionName.data());
    }

    InstanceSupportQueryInfo instance_query_info{file_loader, supported_instance_extensions,
                                                 supported_instance_layers, *this};

    // Check instance support and collect unsupported extensions
    std::vector<std::type_index> unsupported_extensions;
    for (const auto& ext : get_extensions()) {
        auto support_info = ext->query_instance_support(instance_query_info);
        if (!support_info.supported) {
            ext->on_unsupported(support_info.unsupported_reason.empty()
                                    ? "extension instance support check failed."
                                    : support_info.unsupported_reason);
            unsupported_extensions.push_back(typeid(*ext));
        }
    }

    // Remove unsupported extensions
    for (const auto& type_idx : unsupported_extensions) {
        remove_extension(type_idx);
    }

    // -----------------
    // Determine all needed instance extensions
    std::vector<const char*> instance_layer_names;
    std::vector<const char*> instance_extension_names;

    // minimum requirements for Merian
    if (effective_vk_instance_api_version < VK_API_VERSION_1_1) {
        if (!supported_instance_extensions.contains(
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            throw MerianException{
                fmt::format("Merian needs Vulkan 1.1 or {}",
                            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)};
        }
        instance_extension_names.emplace_back(
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }

    // we ignore context extensions here, since we assume that they already do the right checks.
    std::vector<const char*> device_extensions = desired_features.get_required_extensions();
    for (const auto& ext : desired_additional_extensions) {
        device_extensions.emplace_back(ext);
    }
    const auto add_instance_extensions = [&](const auto& self, const char* ext) -> void {
        const ExtensionInfo* const ext_info = get_extension_info(ext);
        if (ext_info == nullptr) {
            throw std::invalid_argument{fmt::format("extension {} unknown", ext)};
        }
        for (const ExtensionInfo* dep : ext_info->dependencies) {
            if (dep->is_instance_extension() &&
                dep->promoted_to_version > effective_vk_instance_api_version) {
                if (supported_instance_extensions.contains(dep->name)) {
                    instance_extension_names.emplace_back(dep->name);
                } else {
                    SPDLOG_WARN("instance extension {} (indirectly) requested but not supported.",
                                dep->name);
                }
            }
            self(self, dep->name);
        }
    };
    for (const auto& ext : device_extensions) {
        add_instance_extensions(add_instance_extensions, ext);
    }

    for (const auto& ext : get_extensions()) {
        auto support_info = ext->query_instance_support(instance_query_info);
        insert_all(instance_layer_names, support_info.required_layers);
        insert_all(instance_extension_names, support_info.required_extensions);
    }
    remove_duplicates(instance_layer_names);
    remove_duplicates(instance_extension_names);

    // -----------------
    // Create instance

    SPDLOG_DEBUG("enabling instance layers: [{}]", fmt::join(instance_layer_names, ", "));
    SPDLOG_DEBUG("enabling instance extensions: [{}]", fmt::join(instance_extension_names, ", "));

    void* p_next = nullptr;
    for (const auto& ext : get_extensions()) {
        p_next = ext->pnext_instance_create_info(p_next);
    }

    vk::ApplicationInfo application_info{
        application_name.c_str(),
        application_vk_version,
        MERIAN_PROJECT_NAME,
        VK_MAKE_VERSION(MERIAN_VERSION_MAJOR, MERIAN_VERSION_MINOR, MERIAN_VERSION_PATCH),
        targeted_vk_api_version,
    };

    vk::InstanceCreateInfo instance_create_info{
        {}, &application_info, instance_layer_names, instance_extension_names, p_next,
    };

    instance = Instance::create(instance_create_info);

    // Must happen before on_instance_created since it requires dynamic loading
    VULKAN_HPP_DEFAULT_DISPATCHER.init(**instance);
    for (const auto& ext : get_extensions()) {
        ext->on_instance_created(instance, *this);
    }
}

Context::DeviceSupportCache Context::select_physical_device(
    uint32_t filter_vendor_id,
    uint32_t filter_device_id,
    std::string filter_device_name,
    const VulkanFeatures& desired_features,
    const std::vector<const char*>& desired_additional_extensions) {
    const std::vector<PhysicalDeviceHandle> physical_devices = instance->get_physical_devices();
    if (physical_devices.empty()) {
        throw MerianException("No vulkan device found!");
    }

    // Check environment variables
    if (const char* env_vendor_id = std::getenv("MERIAN_DEFAULT_FILTER_VENDOR_ID");
        filter_vendor_id == (uint32_t)-1 && (env_vendor_id != nullptr)) {
        filter_vendor_id = std::strtoul(env_vendor_id, nullptr, 10);
    }
    if (const char* env_device_id = std::getenv("MERIAN_DEFAULT_FILTER_DEVICE_ID");
        filter_device_id == (uint32_t)-1 && (env_device_id != nullptr)) {
        filter_device_id = std::strtoul(env_device_id, nullptr, 10);
    }
    if (const char* env_device_name = std::getenv("MERIAN_DEFAULT_FILTER_DEVICE_NAME");
        filter_device_name.empty() && (env_device_name != nullptr)) {
        filter_device_name = env_device_name;
    }

    std::vector<std::pair<PhysicalDeviceHandle, QueueInfo>> matches;
    for (std::size_t i = 0; i < physical_devices.size(); i++) {
        vk::PhysicalDeviceProperties2 props = physical_devices[i]->get_properties();
        SPDLOG_INFO("found {} {}, vendor id: {}, device id: {}, Vulkan: {}.{}.{}",
                    vk::to_string(props.properties.deviceType),
                    props.properties.deviceName.data(), props.properties.vendorID,
                    props.properties.deviceID, VK_API_VERSION_MAJOR(props.properties.apiVersion),
                    VK_API_VERSION_MINOR(props.properties.apiVersion),
                    VK_API_VERSION_PATCH(props.properties.apiVersion));

        if ((filter_vendor_id == (uint32_t)-1 || filter_vendor_id == props.properties.vendorID) &&
            (filter_device_id == (uint32_t)-1 || filter_device_id == props.properties.deviceID) &&
            (filter_device_name == "" || filter_device_name == props.properties.deviceName)) {

            QueueInfo q_info = determine_queues(physical_devices[i]);
            matches.emplace_back(physical_devices[i], std::move(q_info));
        }
    }

    if (matches.empty()) {
        throw std::runtime_error(fmt::format(
            "no vulkan device found with vendor id: {}, device id: {}, device name: {}.",
            filter_vendor_id == (uint32_t)-1 ? "any" : std::to_string(filter_vendor_id),
            filter_device_id == (uint32_t)-1 ? "any" : std::to_string(filter_device_id),
            filter_device_name.empty() ? "any" : filter_device_name));
    }

    std::unordered_map<PhysicalDeviceHandle, DeviceSupportCache> support_cache;
    auto query_support = [&](const std::pair<PhysicalDeviceHandle, QueueInfo>& match)
        -> const DeviceSupportCache& {
        if (auto it = support_cache.find(match.first); it != support_cache.end())
            return it->second;

        DeviceSupportQueryInfo query_info{
            file_loader, match.first, match.second, *this,
            ShaderCompileContext::create(file_loader->get_search_paths(), match.first)};
        auto& cache = support_cache[match.first];
        for (const auto& ext : get_extensions())
            cache.results[ext] = ext->query_device_support(query_info);
        return cache;
    };

    auto rank_device_type = [](vk::PhysicalDeviceType type) -> int {
        if (type == vk::PhysicalDeviceType::eDiscreteGpu)
            return 3;
        if (type == vk::PhysicalDeviceType::eIntegratedGpu)
            return 2;
        if (type == vk::PhysicalDeviceType::eVirtualGpu)
            return 1;
        return 0;
    };

    auto count_extensions_supported =
        [&](const std::pair<PhysicalDeviceHandle, QueueInfo>& match) -> uint32_t {
        uint32_t count = 0;
        for (const auto& ext : desired_additional_extensions)
            count += static_cast<uint32_t>(match.first->extension_supported(ext));
        for (const auto& [_, result] : query_support(match).results)
            for (const auto& ext : result.required_extensions)
                count += static_cast<uint32_t>(match.first->extension_supported(ext));
        return count;
    };

    auto count_features_supported =
        [&](const std::pair<PhysicalDeviceHandle, QueueInfo>& match) -> uint32_t {
        uint32_t count = 0;
        for (const auto& name : desired_features.get_enabled_features())
            count += static_cast<uint32_t>(
                match.first->get_supported_features().get_feature(name));
        for (const auto& [_, result] : query_support(match).results)
            for (const auto& feat : result.required_features)
                count += static_cast<uint32_t>(
                    match.first->get_supported_features().get_feature(feat));
        return count;
    };

    auto best = std::max_element(matches.begin(), matches.end(),
        [&](const auto& a, const auto& b) {
            const vk::PhysicalDeviceProperties& props_a = a.first->get_properties();
            const vk::PhysicalDeviceProperties& props_b = b.first->get_properties();
            if (props_a.deviceType != props_b.deviceType)
                return rank_device_type(props_a.deviceType) < rank_device_type(props_b.deviceType);

            uint32_t ext_a = count_extensions_supported(a);
            uint32_t ext_b = count_extensions_supported(b);
            if (ext_a != ext_b)
                return ext_a < ext_b;

            uint32_t feat_a = count_features_supported(a);
            uint32_t feat_b = count_features_supported(b);
            if (feat_a != feat_b)
                return feat_a < feat_b;

            return a.first->get_supported_extensions().size() <
                   b.first->get_supported_extensions().size();
        });

    physical_device = best->first;
    queue_info = std::move(best->second);

    const vk::PhysicalDeviceProperties& props = physical_device->get_properties();
    const vk::PhysicalDeviceVulkan12Properties& props12 = physical_device->get_properties();

    SPDLOG_INFO("selected physical device {}, vendor id: {}, device id: {}, driver: {}, {}",
                props.deviceName.data(), props.vendorID, props.deviceID,
                vk::to_string(props12.driverID), props12.driverInfo.data());

    auto selected_support = query_support(*best);
    std::vector<std::type_index> unsupported_extensions;
    for (const auto& ext : get_extensions()) {
        const auto& result = selected_support.results.at(ext);
        if (!result.supported) {
            ext->on_unsupported(result.unsupported_reason.empty()
                                    ? "extension device support check failed."
                                    : result.unsupported_reason);
            unsupported_extensions.push_back(typeid(*ext));
        }
    }

    for (const auto& type_idx : unsupported_extensions) {
        remove_extension(type_idx);
    }

    for (const auto& ext : get_extensions()) {
        ext->on_physical_device_selected(physical_device, *this);
    }

    return selected_support;
}

QueueInfo Context::determine_queues(const PhysicalDeviceHandle& physical_device) {
    QueueInfo q_info;

    std::vector<vk::QueueFamilyProperties> queue_family_props =
        physical_device->get_physical_device().getQueueFamilyProperties();

    if (queue_family_props.empty()) {
        throw std::runtime_error{"no queue families available!"};
    }
    SPDLOG_DEBUG("number of queue families available: {}", queue_family_props.size());

    using Flags = vk::QueueFlagBits;

    // We calculate all possible index candidates then sort descending the list to get the best
    // match. (GCT found, GCT extension_accept, additional T found, additional C found, number
    // remaining compute queues, GCT family index, T family index, C family index)
    std::vector<std::tuple<bool, uint32_t, bool, bool, uint32_t, uint32_t, uint32_t, uint32_t>>
        candidates;

#ifndef NDEBUG
    for (std::size_t i = 0; i < queue_family_props.size(); i++) {
        const bool supports_graphics =
            static_cast<bool>(queue_family_props[i].queueFlags & Flags::eGraphics);
        const bool supports_transfer =
            static_cast<bool>(queue_family_props[i].queueFlags & Flags::eTransfer);
        const bool supports_compute =
            static_cast<bool>(queue_family_props[i].queueFlags & Flags::eCompute);
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
                    remaining_queue_count[queue_family_idx_GCT] > 0) {
                    found_GCT = true;
                    remaining_queue_count[queue_family_idx_GCT]--;
                }
                // Prio 2: GCT accepted by extensions
                const auto& extensions = get_extensions();
                uint32_t GCT_accepts =
                    !found_GCT ? 0
                               : std::accumulate(extensions.begin(), extensions.end(), 0,
                                                 [&](uint32_t accum, const auto& ext) {
                                                     return accum + ext->accept_graphics_queue(
                                                                        instance, physical_device,
                                                                        queue_family_idx_GCT);
                                                 });
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

                candidates.emplace_back(found_GCT, GCT_accepts, found_T, found_C,
                                        num_compute_queues, queue_family_idx_GCT,
                                        queue_family_idx_T, queue_family_idx_C);
            }
        }
    }

    // Descending order
    std::sort(candidates.begin(), candidates.end(), std::greater<>());
    auto best = candidates[0];

    const bool found_GCT = std::get<0>(best);
    const bool found_T = std::get<2>(best);
    const bool found_C = std::get<3>(best);

    if (!found_GCT || !found_T || !found_C) {
        SPDLOG_WARN("not all requested queue families found! GCT: {} T: {} C: {}", found_GCT,
                    found_T, found_C);
    }

    q_info.queue_family_idx_GCT = found_GCT ? std::get<5>(best) : -1;
    q_info.queue_family_idx_T = found_T ? std::get<6>(best) : -1;
    q_info.queue_family_idx_C = found_C ? std::get<7>(best) : -1;

    SPDLOG_DEBUG("determined queue families indices: GCT: {} ({}/{} accept votes), T: {} C: {}",
                 q_info.queue_family_idx_GCT, std::get<1>(best),
                 found_GCT ? get_extensions().size() : 0, q_info.queue_family_idx_T,
                 q_info.queue_family_idx_C);

    return q_info;
}

void Context::create_device_and_queues(
    uint32_t preferred_number_compute_queues,
    const VulkanFeatures& desired_features,
    const std::vector<const char*>& desired_additional_extensions,
    const DeviceSupportCache& support_cache) {

    // -------------------------------
    // PREPARE QUEUES

    std::vector<vk::QueueFamilyProperties> queue_family_props =
        physical_device->get_physical_device().getQueueFamilyProperties();
    std::vector<uint32_t> count_per_family(queue_family_props.size());
    uint32_t actual_number_compute_queues = 0;

    if (queue_info.queue_family_idx_GCT >= 0) {
        queue_info.queue_idx_GCT = (int32_t)count_per_family[queue_info.queue_family_idx_GCT]++;
        SPDLOG_DEBUG("queue index GCT: {}", queue_info.queue_idx_GCT);
    }
    if (queue_info.queue_family_idx_T >= 0) {
        queue_info.queue_idx_T = (int32_t)count_per_family[queue_info.queue_family_idx_T]++;
        SPDLOG_DEBUG("queue index T: {}", queue_info.queue_idx_T);
    }
    if (queue_info.queue_family_idx_C >= 0) {
        uint32_t remaining_compute_queues =
            queue_family_props[queue_info.queue_family_idx_C].queueCount -
            count_per_family[queue_info.queue_family_idx_C];
        actual_number_compute_queues =
            std::min(remaining_compute_queues, preferred_number_compute_queues);

        for (uint32_t i = 0; i < actual_number_compute_queues; i++) {
            queue_info.queue_idx_C.emplace_back(count_per_family[queue_info.queue_family_idx_C]++);
        }
        SPDLOG_DEBUG("queue indices C: [{}]", fmt::join(queue_info.queue_idx_C, ", "));
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
    // -------------------------------
    // DEVICE EXTENSIONS
    VulkanFeatures features = desired_features;
    std::vector<const char*> extensions = desired_additional_extensions;

    const uint32_t vk_api_version = VK_HEADER_VERSION_COMPLETE;

    for (const auto& ext : get_extensions()) {
        const auto& support_info = support_cache.results.at(ext);

        insert_all(extensions, support_info.required_extensions);

        for (const auto* feature_name : support_info.required_features)
            features.set_feature(feature_name, true);

        for (const auto* spirv_ext : support_info.required_spirv_extensions)
            insert_all(extensions, get_spirv_extension_requirements(spirv_ext, vk_api_version));

        for (const auto* spirv_cap : support_info.required_spirv_capabilities) {
            insert_all(extensions, get_spirv_capability_extensions(spirv_cap, vk_api_version));
            features.enable_features(get_spirv_capability_features(spirv_cap, vk_api_version));
        }
    }

    // -------------------------------

    std::sort(extensions.begin(), extensions.end());
    remove_duplicates(extensions);

    // Setup p_next for extensions
    // Extensions can enable features of their extensions
    void* extensions_device_create_p_next = nullptr;
    for (const auto& ext : get_extensions()) {
        extensions_device_create_p_next =
            ext->pnext_device_create_info(extensions_device_create_p_next);
    }

    for (const auto& ext : get_extensions()) {
        ext->on_create_device(physical_device, features, extensions);
    }

    device = Device::create(physical_device, features, extensions, queue_create_infos,
                            extensions_device_create_p_next);
    SPDLOG_DEBUG("device created and queues created");

    VULKAN_HPP_DEFAULT_DISPATCHER.init(**device);
    for (const auto& ext : get_extensions()) {
        ext->on_device_created(device, *this);
    }
}

void Context::prepare_file_loader(const ContextCreateInfo& create_info) {
    file_loader = std::make_shared<FileLoader>();

    // Search for merian-shaders includes
    const std::filesystem::path development_headers =
        std::filesystem::path(MERIAN_DEVELOPMENT_INCLUDE_DIR);
    const std::filesystem::path installed_headers =
        std::filesystem::path(FileLoader::install_includedir_name());

    if (FileLoader::exists(development_headers / "merian-shaders")) {
        SPDLOG_DEBUG("found merian-shaders development headers at {}",
                     development_headers.string());
        file_loader->add_search_path(
            std::filesystem::weakly_canonical(development_headers.string()));
    } else if (FileLoader::exists(installed_headers / "merian-shaders")) {
        SPDLOG_DEBUG("found merian-shaders installed at {}", installed_headers.string());
        file_loader->add_search_path(std::filesystem::weakly_canonical(installed_headers.string()));
    } else if (const std::optional<std::filesystem::path> headers =
                   FileLoader::search_cwd_parents("include/merian-shaders");
               headers.has_value()) {
        SPDLOG_DEBUG("found merian-shaders at {}", headers->parent_path().string());
        file_loader->add_search_path(
            std::filesystem::weakly_canonical(headers->parent_path().string()));
    } else {
        SPDLOG_ERROR("merian-shaders header not found! Shader compilers will not work correctly");
    }

    // Add common folders to file loader
    if (const auto portable_prefix = FileLoader::portable_prefix(); portable_prefix) {
        file_loader->add_search_path(*portable_prefix);
    }
    if (const auto install_prefix = FileLoader::install_prefix(); install_prefix) {
        file_loader->add_search_path(*install_prefix);
    }
    file_loader->add_search_path(FileLoader::install_datadir_name());
    file_loader->add_search_path(FileLoader::install_datadir_name() / MERIAN_PROJECT_NAME);
    file_loader->add_search_path(FileLoader::install_includedir_name());

    file_loader->add_search_path(create_info.additional_search_paths);
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
    }
    if (queue_info.queue_family_idx_GCT < 0) {
        return nullptr;
    }
    const auto queue = std::make_shared<Queue>(shared_from_this(), queue_info.queue_family_idx_GCT,
                                               queue_info.queue_idx_GCT);
    queue_GCT = queue;
    return queue;
}

std::shared_ptr<Queue> Context::get_queue_T(const bool fallback) {
    if (queue_info.queue_family_idx_T < 0) {
        if (fallback)
            return get_queue_GCT();
        return nullptr;
    }
    if (!queue_T.expired()) {
        return queue_T.lock();
    }
    const auto queue = std::make_shared<Queue>(shared_from_this(), queue_info.queue_family_idx_T,
                                               queue_info.queue_idx_T);
    queue_T = queue;
    return queue;
}

std::shared_ptr<Queue> Context::get_queue_C(uint32_t index, const bool fallback) {
    assert(fallback || index < queues_C.size());

    if (index < queues_C.size()) {
        if (!queues_C[index].expired()) {
            return queues_C[index].lock();
        }
        const auto queue = std::make_shared<Queue>(
            shared_from_this(), queue_info.queue_family_idx_C, queue_info.queue_idx_C[index]);
        queues_C[index] = queue;
        return queue;
    }
    if (!fallback) {
        // early out, fallback is not allowed
        return nullptr;
    }
    if (!queues_C.empty()) {
        auto unused_queue = std::find_if(queues_C.begin(), queues_C.end(),
                                         [](auto& queue) { return queue.expired(); });
        if (unused_queue == queues_C.end()) {
            // there is no unused queue, use first
            return get_queue_C(0);
        }
        return get_queue_C(unused_queue - queues_C.begin());
    }
    // there are no extra compute queues, maybe at least a graphics queue with compute support
    return get_queue_GCT();
}

std::shared_ptr<CommandPool> Context::get_cmd_pool_GCT() {
    if (!cmd_pool_GCT.expired()) {
        return cmd_pool_GCT.lock();
    }
    const auto cmd = std::make_shared<CommandPool>(get_queue_GCT());
    cmd_pool_GCT = cmd;
    return cmd;
}

std::shared_ptr<CommandPool> Context::get_cmd_pool_T() {
    if (!cmd_pool_T.expired()) {
        return cmd_pool_T.lock();
    }
    const auto cmd = std::make_shared<CommandPool>(get_queue_T());
    cmd_pool_T = cmd;
    return cmd;
}

std::shared_ptr<CommandPool> Context::get_cmd_pool_C() {
    if (!cmd_pool_C.expired()) {
        return cmd_pool_C.lock();
    }
    const auto cmd = std::make_shared<CommandPool>(get_queue_C());
    cmd_pool_C = cmd;
    return cmd;
}

const InstanceHandle& Context::get_instance() const {
    return instance;
}

const PhysicalDeviceHandle& Context::get_physical_device() const {
    return physical_device;
}

const DeviceHandle& Context::get_device() const {
    return device;
}

const FileLoaderHandle& Context::get_file_loader() const {
    return file_loader;
}

const QueueInfo& Context::get_queue_info() const {
    return queue_info;
}

const ShaderCompileContextHandle& Context::get_shader_compile_context() const {
    return shader_compile_context;
}

} // namespace merian
