#include "merian/vk/context.hpp"
#include "merian/shader/slang_session.hpp"
#include "merian/utils/pointer.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/extension_registry.hpp"

#include <fmt/ranges.h>
#include <numeric>
#include <queue>
#include <spdlog/spdlog.h>
#include <tuple>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace merian {

ContextHandle Context::create(const ContextCreateInfo& create_info) {
    const ContextHandle context = std::shared_ptr<Context>(new Context(create_info));

    for (auto& ext : context->context_extensions) {
        ext.second->on_context_created(context, *context);
    }

    return context;
}

void Context::load_extensions(const std::vector<std::string>& extension_names) {
    std::unordered_set<std::string> loaded_extensions;
    std::queue<std::string> extension_names_to_process;

    // Add user-provided extension names to queue
    for (const auto& ext_name : extension_names) {
        extension_names_to_process.push(ext_name);
    }

    context_extensions.clear();

    while (!extension_names_to_process.empty()) {
        const std::string ext_name = extension_names_to_process.front();
        extension_names_to_process.pop();

        // Skip if already loaded
        if (loaded_extensions.contains(ext_name)) {
            continue;
        }

        // Create extension from registry
        auto ext = ExtensionRegistry::get_instance().create(ext_name);
        if (!ext) {
            SPDLOG_WARN("Extension '{}' not found in registry", ext_name);
            continue;
        }

        SPDLOG_DEBUG("Loading extension: {}", ext_name);
        loaded_extensions.insert(ext_name);
        context_extensions[typeindex_from_pointer(ext)] = ext;

        // Add requested dependencies to queue
        auto requested = ext->request_extensions();
        for (const auto& dep_name : requested) {
            extension_names_to_process.push(dep_name);
        }
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

    for (const auto& ext : context_extensions) {
        ext.second->on_context_initializing(VULKAN_HPP_DEFAULT_DISPATCHER);
    }

    const uint32_t target_vk_api_version = VK_HEADER_VERSION_COMPLETE;
    create_instance(target_vk_api_version, create_info.desired_features,
                    create_info.additional_extensions);

    select_physical_device(create_info.filter_vendor_id, create_info.filter_device_id,
                           create_info.filter_device_name, create_info.desired_features,
                           create_info.additional_extensions);

    for (auto& ext : this->context_extensions) {
        ext.second->on_extension_support_confirmed(*this);
    }

    create_device_and_queues(create_info.preferred_number_compute_queues,
                             create_info.desired_features, create_info.additional_extensions);

    prepare_shader_include_defines();

    prepare_file_loader();

    slang_session = SlangSession::get_or_create(ShaderCompileContext::create(*this));

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
        supported_instance_layers.emplace(instance_layer.layerName);
    }
    for (const auto& instance_ext : vk::enumerateInstanceExtensionProperties()) {
        supported_instance_extensions.emplace(instance_ext.extensionName);
    }

    auto it = context_extensions.begin();
    while (it != context_extensions.end()) {
        InstanceSupportQueryInfo query_info{supported_instance_extensions,
                                             supported_instance_layers, *this};
        auto support_info = it->second->query_instance_support(query_info);
        if (support_info.supported) {
            it++;
        } else {
            it->second->on_unsupported("extension instance support check failed.");
            it = context_extensions.erase(it);
        }
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

    for (auto& ext : context_extensions) {
        InstanceSupportQueryInfo query_info{supported_instance_extensions,
                                             supported_instance_layers, *this};
        auto support_info = ext.second->query_instance_support(query_info);
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
    for (auto& ext : context_extensions) {
        p_next = ext.second->pnext_instance_create_info(p_next);
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
    for (auto& ext : context_extensions) {
        ext.second->on_instance_created(instance);
    }
}

void Context::select_physical_device(
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

    // (device, props, number_of_accept_votes)
    std::vector<std::tuple<PhysicalDeviceHandle, QueueInfo, uint32_t, uint32_t, uint32_t>> matches;
    for (std::size_t i = 0; i < physical_devices.size(); i++) {
        vk::PhysicalDeviceProperties2 props = physical_devices[i]->get_properties();
        SPDLOG_INFO("found physical device {}, vendor id: {}, device id: {}, Vulkan: {}.{}.{}",
                    props.properties.deviceName.data(), props.properties.vendorID,
                    props.properties.deviceID, VK_API_VERSION_MAJOR(props.properties.apiVersion),
                    VK_API_VERSION_MINOR(props.properties.apiVersion),
                    VK_API_VERSION_PATCH(props.properties.apiVersion));

        if ((filter_vendor_id == (uint32_t)-1 || filter_vendor_id == props.properties.vendorID) &&
            (filter_device_id == (uint32_t)-1 || filter_device_id == props.properties.deviceID) &&
            (filter_device_name == "" || filter_device_name == props.properties.deviceName)) {

            QueueInfo q_info = determine_queues(physical_devices[i]);

            uint32_t context_extensions_supported = 0;
            for (const auto& ext : context_extensions) {
                DeviceSupportQueryInfo query_info{physical_devices[i], q_info, *this};
                auto support_info = ext.second->query_device_support(query_info);
                context_extensions_supported += static_cast<uint32_t>(support_info.supported);
            }

            uint32_t extensions_supported = 0;
            uint32_t extensions_total = 0;
            for (const auto& ext : desired_additional_extensions) {
                extensions_supported +=
                    static_cast<uint32_t>(physical_devices[i]->extension_supported(ext));
                extensions_total++;
            }

            uint32_t features_supported = 0;
            uint32_t features_total = 0;
            for (const auto& feature_name : desired_features.get_enabled_features()) {
                features_supported += static_cast<uint32_t>(
                    physical_devices[i]->get_supported_features().get_feature(feature_name));
                features_total++;
            }

            SPDLOG_DEBUG(
                "device supports {}/{} context extensions, {}/{} requested additional extensions, "
                "{}/{} requested features.",
                context_extensions_supported, context_extensions.size(), extensions_supported,
                extensions_total, features_supported, features_total);

            matches.emplace_back(physical_devices[i], std::move(q_info),
                                 context_extensions_supported, extensions_supported,
                                 features_supported);
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
              [](std::tuple<PhysicalDeviceHandle, QueueInfo, uint32_t, uint32_t, uint32_t>& a,
                 std::tuple<PhysicalDeviceHandle, QueueInfo, uint32_t, uint32_t, uint32_t>& b) {
                  // context extension support
                  if (std::get<2>(a) != std::get<2>(b)) {
                      return std::get<2>(a) < std::get<2>(b);
                  }

                  const vk::PhysicalDeviceProperties& props_a = std::get<0>(a)->get_properties();
                  const vk::PhysicalDeviceProperties& props_b = std::get<0>(b)->get_properties();
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
                  // extension support
                  if (std::get<3>(a) != std::get<3>(b)) {
                      return std::get<3>(a) < std::get<3>(b);
                  }
                  // feature support
                  if (std::get<4>(a) != std::get<4>(b)) {
                      return std::get<4>(a) < std::get<4>(b);
                  }

                  return std::get<0>(a)->get_supported_extensions().size() <
                         std::get<0>(b)->get_supported_extensions().size();

                  return false;
              });

    physical_device = std::get<0>(matches.back());
    queue_info = std::move(std::get<1>(matches.back()));

    const vk::PhysicalDeviceProperties& props = physical_device->get_properties();
    const vk::PhysicalDeviceVulkan12Properties& props12 = physical_device->get_properties();

    SPDLOG_INFO("selected physical device {}, vendor id: {}, device id: {}, driver: {}, {}",
                props.deviceName.data(), props.vendorID, props.deviceID,
                vk::to_string(props12.driverID), props12.driverInfo.data());

    auto it = context_extensions.begin();
    while (it != context_extensions.end()) {
        DeviceSupportQueryInfo query_info{physical_device, queue_info, *this};
        auto support_info = it->second->query_device_support(query_info);
        if (support_info.supported) {
            it++;
        } else {
            it->second->on_unsupported("extension device support check failed.");
            it = context_extensions.erase(it);
        }
    }

    for (auto& ext : context_extensions) {
        ext.second->on_physical_device_selected(physical_device);
    }
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
                uint32_t GCT_accepts =
                    !found_GCT
                        ? 0
                        : std::accumulate(context_extensions.begin(), context_extensions.end(), 0,
                                          [&](uint32_t accum, auto& ext) {
                                              return accum + ext.second->accept_graphics_queue(
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
                 found_GCT ? context_extensions.size() : 0, q_info.queue_family_idx_T,
                 q_info.queue_family_idx_C);

    return q_info;
}

void Context::create_device_and_queues(
    uint32_t preferred_number_compute_queues,
    const VulkanFeatures& desired_features,
    const std::vector<const char*>& desired_additional_extensions) {

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

    for (const auto& ext : context_extensions) {
        DeviceSupportQueryInfo query_info{physical_device, queue_info, *this};
        auto support_info = ext.second->query_device_support(query_info);
        insert_all(extensions, support_info.required_extensions);
        for (const auto* feature_name : support_info.required_features) {
            features.set_feature(feature_name, true);
        }
    }

    // -------------------------------

    // Setup p_next for extensions
    // Extensions can enable features of their extensions
    void* extensions_device_create_p_next = nullptr;
    for (auto& ext : context_extensions) {
        extensions_device_create_p_next =
            ext.second->pnext_device_create_info(extensions_device_create_p_next);
    }

    for (auto& ext : context_extensions) {
        ext.second->on_create_device(physical_device, features, extensions);
    }

    device = Device::create(physical_device, features, extensions, queue_create_infos,
                            extensions_device_create_p_next);
    SPDLOG_DEBUG("device created and queues created");

    VULKAN_HPP_DEFAULT_DISPATCHER.init(**device);
    for (auto& ext : context_extensions) {
        ext.second->on_device_created(device);
    }
}

void Context::prepare_shader_include_defines() {
    // search merian-shaders
    const std::filesystem::path development_headers =
        std::filesystem::path(MERIAN_DEVELOPMENT_INCLUDE_DIR);
    const std::filesystem::path installed_headers =
        std::filesystem::path(FileLoader::install_includedir_name());

    if (FileLoader::exists(development_headers / "merian-shaders")) {
        SPDLOG_DEBUG("found merian-shaders development headers headers at {}",
                     development_headers.string());
        default_shader_include_paths.emplace_back(
            std::filesystem::weakly_canonical(development_headers.string()));
    } else if (FileLoader::exists(installed_headers / "merian-shaders")) {
        SPDLOG_DEBUG("found merian-shaders installed at {}", installed_headers.string());
        default_shader_include_paths.emplace_back(
            std::filesystem::weakly_canonical(installed_headers.string()));
    } else if (const std::optional<std::filesystem::path> headers =
                   FileLoader::search_cwd_parents("include/merian-shaders");
               headers.has_value()) {
        SPDLOG_DEBUG("found merian-shaders at {}", headers->parent_path().string());
        default_shader_include_paths.emplace_back(
            std::filesystem::weakly_canonical(headers->parent_path().string()));
    } else {
        SPDLOG_ERROR("merian-shaders header not found! Shader compilers will not work correctly");
    }
}

void Context::prepare_file_loader() {
    // add those first => prefer development headers.
    file_loader.add_search_path(default_shader_include_paths);

    // add common folders to file loader
    if (const auto portable_prefix = FileLoader::portable_prefix(); portable_prefix) {
        file_loader.add_search_path(*portable_prefix);
    }
    if (const auto install_prefix = FileLoader::install_prefix(); install_prefix) {
        file_loader.add_search_path(*install_prefix);
    }
    file_loader.add_search_path(FileLoader::install_datadir_name());
    file_loader.add_search_path(FileLoader::install_datadir_name() / MERIAN_PROJECT_NAME);
    file_loader.add_search_path(FileLoader::install_includedir_name());
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

const std::vector<std::filesystem::path>& Context::get_default_shader_include_paths() const {
    return default_shader_include_paths;
}

const SlangSessionHandle& Context::get_slang_session() const {
    return slang_session;
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

FileLoader& Context::get_file_loader() {
    return file_loader;
}

} // namespace merian
