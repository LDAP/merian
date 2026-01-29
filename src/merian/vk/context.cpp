#include "merian/vk/context.hpp"
#include "merian/shader/slang_session.hpp"
#include "merian/utils/pointer.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/vk/utils/vulkan_spirv.hpp"

#include <fmt/ranges.h>
#include <numeric>
#include <spdlog/spdlog.h>
#include <tuple>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace merian {

ContextHandle
Context::create(const VulkanFeatures& desired_features,
                const std::vector<const char*>& desired_additional_extensions,
                const std::vector<std::shared_ptr<ContextExtension>>& desired_context_extensions,
                const std::string& application_name,
                const uint32_t application_vk_version,
                const uint32_t preffered_number_compute_queues,
                const uint32_t vk_api_version,
                const uint32_t filter_vendor_id,
                const uint32_t filter_device_id,
                const std::string& filter_device_name) {

    // call constructor manually since its private for make_shared...
    const ContextHandle context = std::shared_ptr<Context>(
        new Context(desired_features, desired_additional_extensions, desired_context_extensions,
                    application_name, application_vk_version, preffered_number_compute_queues,
                    vk_api_version, filter_vendor_id, filter_device_id, filter_device_name));

    for (auto& ext : context->context_extensions) {
        ext.second->on_context_created(context, *context);
    }

    return context;
}

Context::Context(const VulkanFeatures& desired_features,
                 const std::vector<const char*>& desired_additional_extensions,
                 const std::vector<std::shared_ptr<ContextExtension>>& desired_context_extensions,
                 const std::string& application_name,
                 const uint32_t application_vk_version,
                 const uint32_t preffered_number_compute_queues,
                 const uint32_t vk_api_version,
                 const uint32_t filter_vendor_id,
                 const uint32_t filter_device_id,
                 const std::string& filter_device_name)
    : application_name(application_name), application_vk_version(application_vk_version) {
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

    // Init dynamic loader
    VULKAN_HPP_DEFAULT_DISPATCHER.init();

    SPDLOG_DEBUG("supplied extensions:");
    for (const auto& ext : desired_context_extensions) {
        SPDLOG_DEBUG("{}", ext->name);
        if (this->context_extensions.contains(typeindex_from_pointer(ext))) {
            throw std::runtime_error{"A extension type can only be added once."};
        }
        this->context_extensions[typeindex_from_pointer(ext)] = ext;
    }

    for (const auto& ext : context_extensions) {
        ext.second->on_context_initializing(*this);
    }

    create_instance(vk_api_version, desired_features, desired_additional_extensions);

    select_physical_device(filter_vendor_id, filter_device_id, filter_device_name, desired_features,
                           desired_additional_extensions);

    for (auto& ext : this->context_extensions) {
        ext.second->on_extension_support_confirmed(*this);
    }

    create_device_and_queues(preffered_number_compute_queues, desired_features,
                             desired_additional_extensions);

    prepare_shader_include_defines();

    prepare_file_loader();

    slang_session = SlangSession::get_or_create(ShaderCompileContext::create(*this));

    SPDLOG_INFO("context ready. (took: {})", format_duration(sw.nanos()));
}

Context::~Context() {
    SPDLOG_INFO("context destroyed");
}

void Context::create_instance(const uint32_t vk_api_version,
                              const VulkanFeatures& desired_features,
                              const std::vector<const char*>& desired_additional_extensions) {
    std::unordered_set<std::string> supported_instance_layers;
    std::unordered_set<std::string> supported_instance_extensions;
    for (const auto& instance_layer : vk::enumerateInstanceLayerProperties()) {
        supported_instance_layers.emplace(instance_layer.layerName);
    }
    for (const auto& instance_ext : vk::enumerateInstanceExtensionProperties()) {
        supported_instance_extensions.emplace(instance_ext.extensionName);
    }

    // -----------------
    // Remove unsupported extensions

    auto it = context_extensions.begin();
    while (it != context_extensions.end()) {
        if (it->second->extension_supported(supported_instance_extensions,
                                            supported_instance_layers)) {
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

    // we ignore context extensions here, since we assume that they already do the right checks.
    std::vector<const char*> device_extensions = desired_features.get_required_extensions(vk_api_version);
    for (const auto& ext : desired_additional_extensions) {
        device_extensions.emplace_back(ext);
    }
    const auto add_instance_extensions = [&](const auto& self, const char* ext) -> void {
        for (const ExtensionInfo* dep : get_extension_info(ext)->dependencies) {
            SPDLOG_INFO(dep->name);
            if (dep->is_instance_extension() && dep->promoted_to_version > vk_api_version) {
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

    // from context extensions:
    for (auto& ext : context_extensions) {
        insert_all(instance_layer_names,
                   ext.second->enable_instance_layer_names(supported_instance_layers));
        insert_all(instance_extension_names,
                   ext.second->enable_instance_extension_names(supported_instance_extensions));
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
        vk_api_version,
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
                context_extensions_supported += static_cast<uint32_t>(
                    ext.second->extension_supported(physical_devices[i], q_info));
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
            for (const auto& feature_struct_name : desired_features.get_feature_struct_names()) {
                for (const auto& feature :
                     desired_features.get_feature_names(feature_struct_name)) {
                    if (desired_features.get_feature(feature_struct_name, feature)) {
                        features_supported += static_cast<uint32_t>(
                            physical_devices[i]->get_supported_features().get_feature(
                                feature_struct_name, feature));
                        features_total++;
                    }
                }
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
        if (it->second->extension_supported(physical_device, queue_info)) {
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
        insert_all(extensions, ext.second->enable_device_extension_names(physical_device));
    }

    // FEATURES
    for (const auto& ext : context_extensions) {
        features.enable_features(ext.second->enable_device_features(physical_device));
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

bool spirv_extension_supported(const DeviceHandle& device, const char* extension) {
    const auto device_extension_deps = get_spirv_extension_requirements(
        extension, device->get_physical_device()->get_instance()->get_vk_api_version());
    bool all_enabled = true;
    for (const auto& dep : device_extension_deps) {
        all_enabled &= device->extension_enabled(dep);
    }
    return all_enabled;
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

    // add macro definitions from context extensions and enabled instance and device extensions.
    for (const auto& ext : context_extensions) {
        const auto extension_macro_definitions = ext.second->shader_macro_definitions();
        default_shader_macro_definitions.insert(extension_macro_definitions.begin(),
                                                extension_macro_definitions.end());
        default_shader_macro_definitions.emplace("MERIAN_CONTEXT_EXT_ENABLED_" + ext.second->name,
                                                 "1");
    }
    const std::string instance_ext_define_prefix = "MERIAN_INSTANCE_EXT_ENABLED_";
    for (const auto& ext : instance->get_enabled_extensions()) {
        default_shader_macro_definitions.emplace(instance_ext_define_prefix + ext, "1");
    }
    const std::string device_ext_define_prefix = "MERIAN_DEVICE_EXT_ENABLED_";
    for (const auto& ext : device->get_enabled_extensions()) {
        default_shader_macro_definitions.emplace(device_ext_define_prefix + ext, "1");
    }
    const std::string spirv_ext_define_prefix = "MERIAN_SPIRV_EXT_SUPPORTED_";
    for (const auto& ext : get_spirv_extensions()) {
        if (spirv_extension_supported(device, ext)) {
            default_shader_macro_definitions.emplace(spirv_ext_define_prefix + ext, "1");
        }
    }
    const std::string spirv_cap_define_prefix = "MERIAN_SPIRV_CAP_SUPPORTED_";
    for (const auto& cap : get_spirv_capabilities()) {
        if (is_spirv_capability_supported(cap, instance->get_vk_api_version(),
                                          device->get_enabled_features(),
                                          physical_device->get_properties())) {
            default_shader_macro_definitions.emplace(spirv_cap_define_prefix + cap, "1");
        }
    }

#ifndef NDEBUG
    std::vector<std::string> string_defines;
    string_defines.reserve(default_shader_macro_definitions.size());
    for (const auto& def : default_shader_macro_definitions) {
        string_defines.emplace_back(def.first + "=" + def.second);
    }
    SPDLOG_DEBUG("shader defines: {}", fmt::join(string_defines, "\n"));
#endif
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

const std::map<std::string, std::string>& Context::get_default_shader_macro_definitions() const {
    return default_shader_macro_definitions;
}

const SlangSessionHandle& Context::get_slang_session() const {
    return slang_session;
}

const InstanceHandle& Context::get_instance() {
    return instance;
}

const PhysicalDeviceHandle& Context::get_physical_device() {
    return physical_device;
}

const DeviceHandle& Context::get_device() {
    return device;
}

const uint32_t& Context::get_vk_api_version() const {
    return instance->get_vk_api_version();
}

FileLoader& Context::get_file_loader() {
    return file_loader;
}

} // namespace merian
