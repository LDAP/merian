#include "merian/vk/context.hpp"
#include "merian/plugin/plugins.hpp"
#include "merian/utils/pointer.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/extension_registry.hpp"
#include "merian/vk/utils/vulkan_extensions.hpp"
#include "merian/vk/utils/vulkan_spirv.hpp"

#include <algorithm>
#include <bit>
#include <fmt/ranges.h>
#include <queue>
#include <ranges>
#include <spdlog/spdlog.h>
#include <tuple>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace merian {

struct Context::DeviceSupportCache {
    std::unordered_map<std::shared_ptr<ContextExtension>, DeviceSupportInfo> results;
};

struct Context::FeatureExtensionCheckResult {
    VulkanFeatures features;
    std::vector<const char*> extensions;
    std::vector<const char*> missing_instance_extensions;
};

ContextHandle Context::create(const ContextCreateInfo& create_info) {
    // Let plugins register their context extensions before any extension is loaded.
    ExtensionRegistry::get_instance().load_from_plugins();

    const ContextHandle context = std::shared_ptr<Context>(new Context(create_info));

    for (const auto& ext : context->get_extensions()) {
        ext->on_context_created(context, *context);
    }

    return context;
}

int ExtensionContainer::get_extension_priority(const std::shared_ptr<ContextExtension>& ext,
                                               const std::type_index& interface_type) const {
    const auto& reg = ExtensionRegistry::get_instance();
    return reg.get_priority(reg.get_name(ext), interface_type);
}

void ExtensionContainer::add_extension(const std::shared_ptr<ContextExtension>& extension) {
    const std::type_index type_idx = typeindex_from_pointer(extension);
    context_extensions[type_idx] = extension;
    ordered_extensions.push_back(extension);
}

void ExtensionContainer::remove_extension(const std::type_index& type) {
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

    // Queue auto-load extensions first, then user-requested extensions
    for (const auto& ext_name : ExtensionRegistry::get_instance().get_auto_load_names()) {
        to_process.push(ext_name);
    }
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

void Context::determine_instance_extension_layer_support(const uint32_t targeted_vk_api_version) {
    const uint32_t effective_vk_instance_api_version =
        std::min(targeted_vk_api_version, Instance::get_instance_vk_api_version());

    SPDLOG_TRACE("checking instance layer support...");
    for (const auto& instance_layer : vk::enumerateInstanceLayerProperties()) {
        supported_instance_layers.emplace(instance_layer.layerName.data());
        SPDLOG_TRACE("{} supported", instance_layer.layerName.data());
    }

    std::unordered_set<std::string> all_instance_extensions;
    for (const auto& ext_props : vk::enumerateInstanceExtensionProperties()) {
        all_instance_extensions.insert(ext_props.extensionName.data());
    }

    const auto check_extension_recurse = [&](const auto& self,
                                             const char* ext) -> const InstanceExtensionSupport& {
        const auto it = supported_instance_extensions.find(ext);
        if (it != supported_instance_extensions.end()) {
            return it->second;
        }

        // can happen as part of a dependency
        if (!all_instance_extensions.contains(ext)) {
            const auto [ins_it, inserted] = supported_instance_extensions.emplace(
                ext, InstanceExtensionSupport{
                         false, "the extension is not advertised as supported", {}});
            return ins_it->second;
        }

        const ExtensionInfo* ext_info = get_extension_info(ext);
        if (ext_info == nullptr) {
            const auto [ins_it, inserted] = supported_instance_extensions.emplace(
                ext, InstanceExtensionSupport{true, "extension unknown", {}});
            return ins_it->second;
        }
        assert(ext_info->is_instance_extension());

        if (ext_info->dependencies.empty()) {
            const auto [ins_it, inserted] =
                supported_instance_extensions.emplace(ext, InstanceExtensionSupport{true, "", {}});
            return ins_it->second;
        }

        // Check dependency branches in order of increasing dependent extension count
        std::vector<ExtensionDependency> dependencies(ext_info->dependencies.begin(),
                                                      ext_info->dependencies.end());
        std::sort(dependencies.begin(), dependencies.end(),
                  [](const ExtensionDependency& a, const ExtensionDependency& b) -> bool {
                      return a.required_extensions.size() < b.required_extensions.size();
                  });

        std::unordered_set<std::string> unsupported_reasons;
        for (const ExtensionDependency& dep : dependencies) {
            if (dep.required_version > effective_vk_instance_api_version) {
                unsupported_reasons.insert(
                    fmt::format("Vulkan API version {} is required",
                                format_vk_api_version(dep.required_version)));
                continue;
            }

            InstanceExtensionSupport support{true};
            for (const ExtensionInfo* req_dep_ext : dep.required_extensions) {
                const InstanceExtensionSupport& dep_support = self(self, req_dep_ext->name);
                assert(req_dep_ext->is_instance_extension());
                if (dep_support.supported) {
                    support.dependencies.insert(req_dep_ext->name);
                    support.dependencies.insert(dep_support.dependencies.begin(),
                                                dep_support.dependencies.end());
                } else {
                    support.supported = false;
                    unsupported_reasons.insert(
                        fmt::format("requires {}, which is unsupported because {}",
                                    req_dep_ext->name, dep_support.info));
                    break;
                }
            }
            if (support.supported) {
                const auto [ins_it, inserted] =
                    supported_instance_extensions.emplace(ext, std::move(support));
                return ins_it->second;
            }
        }

        const auto [ins_it, inserted] = supported_instance_extensions.emplace(
            ext, InstanceExtensionSupport{
                     false, fmt::format("{}", fmt::join(unsupported_reasons, " or ")), {}});
        return ins_it->second;
    };

    SPDLOG_TRACE("checking instance extension support...");
    [[maybe_unused]] const auto format_support =
        [](const InstanceExtensionSupport& support) -> std::string {
        std::string support_str = support.supported ? "supported" : "unsupported";
        if (!support.info.empty()) {
            support_str += fmt::format(" ({})", support.info);
        }
        if (!support.dependencies.empty()) {
            support_str +=
                fmt::format(" dependencies: [{}]", fmt::join(support.dependencies, ", "));
        }
        return support_str;
    };
    for (const auto& ext_props : vk::enumerateInstanceExtensionProperties()) {
        const char* const ext = ext_props.extensionName.data();
        [[maybe_unused]] const InstanceExtensionSupport& support =
            check_extension_recurse(check_extension_recurse, ext);
        SPDLOG_TRACE("{} {}", ext, format_support(support));
    }
}

Context::Context(const ContextCreateInfo& create_info)
    : application_name(create_info.application_name),
      application_vk_version(create_info.application_vk_version) {
    merian::Stopwatch sw;

    uint32_t target_vk_api_version = VK_HEADER_VERSION_COMPLETE;
    const char* env_target_version_str = std::getenv("MERIAN_TARGET_VK_API_VERSION");
    if (env_target_version_str != nullptr) {
        const uint32_t env_target_version = parse_vk_api_version(env_target_version_str);
        if (VK_API_VERSION_1_0 <= env_target_version &&
            env_target_version <= VK_HEADER_VERSION_COMPLETE) {
            target_vk_api_version = env_target_version;
        } else {
            SPDLOG_ERROR("MERIAN_TARGET_VK_API_VERSION must be between {} and {}.",
                         format_vk_api_version(VK_API_VERSION_1_0),
                         format_vk_api_version(VK_HEADER_VERSION_COMPLETE));
        }
    }

    SPDLOG_INFO("\n\n\
__  __ ___ ___ ___   _   _  _ \n\
|  \\/  | __| _ \\_ _| /_\\ | \\| |\n\
| |\\/| | _||   /| | / _ \\| .` |\n\
|_|  |_|___|_|_\\___/_/ \\_\\_|\\_|\n\n\
Version: {}\n\n\
Vulkan-Headers Version: {}\n\
Target API Version: {}\n\
\n\n",
                MERIAN_VERSION, format_vk_api_version(VK_HEADER_VERSION_COMPLETE),
                format_vk_api_version(target_vk_api_version));
    SPDLOG_INFO("context initializing...");

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

    determine_instance_extension_layer_support(target_vk_api_version);

    bool did_recreate = false;
    Context::FeatureExtensionCheckResult feat_ext_result;
    Context::DeviceSupportCache support_cache;
    do {
        did_recreate = !feat_ext_result.missing_instance_extensions.empty();
        if (did_recreate) {
            SPDLOG_WARN("recreating instance to enable missing extensions");
        }

        std::vector<const char*> additional_instance_extensions = create_info.instance_extensions;
        insert_all(additional_instance_extensions, feat_ext_result.missing_instance_extensions);
        create_instance(target_vk_api_version, additional_instance_extensions,
                        create_info.instance_layers);

        support_cache =
            select_physical_device(create_info.filter_vendor_id, create_info.filter_device_id,
                                   create_info.filter_device_name, create_info.features,
                                   create_info.device_extensions, create_info.queues);

        feat_ext_result = determine_features_extensions(
            create_info.features, create_info.device_extensions, support_cache);

    } while (!did_recreate && !feat_ext_result.missing_instance_extensions.empty());

    {
        std::vector<std::type_index> unsupported_extensions;
        for (const auto& ext : get_extensions()) {
            const auto& result = support_cache.results.at(ext);
            if (!result.supported) {
                ext->on_unsupported(result.unsupported_reason.empty()
                                        ? "extension device support check failed."
                                        : result.unsupported_reason);
                unsupported_extensions.push_back(typeindex_from_pointer(ext));
            }
        }
        for (const auto& type_idx : unsupported_extensions) {
            remove_extension(type_idx);
        }

        for (const auto& ext : get_extensions()) {
            ext->on_physical_device_selected(physical_device, *this);
        }
    }

    create_device_and_queues(create_info.queues, feat_ext_result.features,
                             feat_ext_result.extensions);

    shader_compile_context = ShaderCompileContext::create(file_loader->get_search_paths(), device);

    SPDLOG_INFO("context ready. (took: {})", format_duration(sw.nanos()));
}

Context::~Context() {
    SPDLOG_INFO("context destroyed");
}

void Context::create_instance(const uint32_t targeted_vk_api_version,
                              const std::vector<const char*>& desired_additional_extensions,
                              const std::vector<const char*>& desired_additional_layers) {
    const uint32_t effective_vk_instance_api_version =
        std::min(targeted_vk_api_version, Instance::get_instance_vk_api_version());

    std::unordered_set<std::string> supported_instance_extension_names;
    for (const auto& [name, support] : supported_instance_extensions) {
        if (support.supported) {
            supported_instance_extension_names.insert(name);
        }
    }

    // -----------------
    // Determine all needed instance extensions
    std::vector<const char*> instance_layer_names;
    std::vector<const char*> instance_extension_names;

    // minimum requirements for Merian
    if (effective_vk_instance_api_version < VK_API_VERSION_1_1) {
        if (!supported_instance_extension_names.contains(
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            throw MerianException{fmt::format(
                "Merian needs Vulkan {} or {}", format_vk_api_version(VK_API_VERSION_1_1),
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)};
        }
        instance_extension_names.emplace_back(
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }

    // User (and auto added extension from context)
    {
        for (const char* layer : desired_additional_layers) {
            if (supported_instance_layers.contains(layer)) {
                instance_layer_names.push_back(layer);
            } else {
                SPDLOG_WARN("instance layer {} requested but not available", layer);
            }
        }
        for (const char* ext : desired_additional_extensions) {
            if (supported_instance_extension_names.contains(ext)) {
                instance_extension_names.push_back(ext);
            } else {
                SPDLOG_WARN("instance extension {} requested but not available", ext);
            }
        }
    }

    // Extensions
    {
        InstanceSupportQueryInfo instance_query_info{
            file_loader, supported_instance_extension_names, supported_instance_layers, *this};

        // Check instance support and collect extensions, layers then remove unsupported extensions.
        std::vector<std::type_index> unsupported_extensions;
        for (const auto& ext : get_extensions()) {
            auto support_info = ext->query_instance_support(instance_query_info);
            if (support_info.supported) {
                insert_all(instance_layer_names, support_info.required_layers);
                insert_all(instance_extension_names, support_info.required_extensions);
            } else {
                ext->on_unsupported(support_info.unsupported_reason.empty()
                                        ? "extension instance support check failed."
                                        : support_info.unsupported_reason);
                unsupported_extensions.push_back(typeindex_from_pointer(ext));
            }
        }
        // Remove unsupported extensions
        for (const auto& type_idx : unsupported_extensions) {
            remove_extension(type_idx);
        }
    }

    // ------------------------
    // Add dependencies and sanitize
    {
        for (uint32_t i = 0; i < instance_extension_names.size();) {
            const char* ext = instance_extension_names[i];
            const auto support_it = supported_instance_extensions.find(ext);
            if (support_it == supported_instance_extensions.end()) {
                SPDLOG_WARN("removed instance extension {} that is not advertised as supported. ",
                            ext);
                std::swap(instance_extension_names[i], instance_extension_names.back());
                instance_extension_names.pop_back();
                continue;
            }
            if (!support_it->second.supported) {
                SPDLOG_WARN("removed instance extension {} that is unsupported because {}. ", ext,
                            support_it->second.info);
                std::swap(instance_extension_names[i], instance_extension_names.back());
                instance_extension_names.pop_back();
                continue;
            }
            const ExtensionInfo* ext_info = get_extension_info(ext);
            if (ext_info != nullptr &&
                ext_info->promoted_to_version <= effective_vk_instance_api_version) {
                SPDLOG_DEBUG(
                    "removed instance extension {} that is already provided by API version", ext);
                std::swap(instance_extension_names[i], instance_extension_names.back());
                instance_extension_names.pop_back();
                continue;
            }
            if (ext_info != nullptr && ext_info->deprecated_by != nullptr &&
                supported_instance_extension_names.contains(ext_info->deprecated_by->name)) {
                SPDLOG_DEBUG("removed deprecated instance extension {} and replaced with {}", ext,
                             ext_info->deprecated_by->name);
                std::swap(instance_extension_names[i], instance_extension_names.back());
                instance_extension_names.pop_back();
                continue;
            }

            insert_all(instance_extension_names, support_it->second.dependencies);
            i++;
        }

        sort_and_remove_duplicates(instance_layer_names);
        sort_and_remove_duplicates(instance_extension_names);
    }

    // -----------------
    // Create instance

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

    {
        const InstanceSupportQueryInfo support_info{file_loader, supported_instance_extension_names,
                                                    supported_instance_layers, *this};
        for (const auto& ext : get_extensions()) {
            ext->on_create_instance(support_info, application_info, instance_layer_names,
                                    instance_extension_names);
        }
    }

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

Context::DeviceSupportCache
Context::select_physical_device(uint32_t filter_vendor_id,
                                uint32_t filter_device_id,
                                std::string filter_device_name,
                                const VulkanFeatures& desired_additional_features,
                                const std::vector<const char*>& desired_additional_extensions,
                                const std::vector<QueueRequest>& queue_requests) {
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

    std::vector<std::pair<PhysicalDeviceHandle, QueueAssignment>> matches;
    uint32_t rejected_for_queues = 0;
    for (std::size_t i = 0; i < physical_devices.size(); i++) {
        vk::PhysicalDeviceProperties2 props = physical_devices[i]->get_properties();
        SPDLOG_INFO("found {} {}, vendor id: {}, device id: {}, Vulkan: {}.{}.{}",
                    vk::to_string(props.properties.deviceType), props.properties.deviceName.data(),
                    props.properties.vendorID, props.properties.deviceID,
                    VK_API_VERSION_MAJOR(props.properties.apiVersion),
                    VK_API_VERSION_MINOR(props.properties.apiVersion),
                    VK_API_VERSION_PATCH(props.properties.apiVersion));

        if ((filter_vendor_id == (uint32_t)-1 || filter_vendor_id == props.properties.vendorID) &&
            (filter_device_id == (uint32_t)-1 || filter_device_id == props.properties.deviceID) &&
            (filter_device_name == "" || filter_device_name == props.properties.deviceName)) {

            std::optional<QueueAssignment> queues = determine_queues(
                physical_devices[i], collect_queue_requests(physical_devices[i], queue_requests));
            if (queues) {
                matches.emplace_back(physical_devices[i], std::move(*queues));
            } else {
                rejected_for_queues++;
                SPDLOG_INFO("skipping {}: cannot provide the requested queues",
                            props.properties.deviceName.data());
            }
        }
    }

    if (matches.empty()) {
        if (rejected_for_queues > 0) {
            throw std::runtime_error{
                fmt::format("none of the {} matching vulkan devices can provide the required "
                            "queues.",
                            rejected_for_queues)};
        }
        throw std::runtime_error(fmt::format(
            "no vulkan device found with vendor id: {}, device id: {}, device name: {}.",
            filter_vendor_id == (uint32_t)-1 ? "any" : std::to_string(filter_vendor_id),
            filter_device_id == (uint32_t)-1 ? "any" : std::to_string(filter_device_id),
            filter_device_name.empty() ? "any" : filter_device_name));
    }

    std::unordered_map<PhysicalDeviceHandle, DeviceSupportCache> support_cache;
    auto query_support = [&](const std::pair<PhysicalDeviceHandle, QueueAssignment>& match)
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
        [&](const std::pair<PhysicalDeviceHandle, QueueAssignment>& match) -> uint32_t {
        uint32_t count = 0;
        for (const auto& ext : desired_additional_extensions)
            count += static_cast<uint32_t>(match.first->extension_supported(ext));
        for (const auto& [_, result] : query_support(match).results)
            for (const auto& ext : result.required_extensions)
                count += static_cast<uint32_t>(match.first->extension_supported(ext));
        return count;
    };

    auto count_features_supported =
        [&](const std::pair<PhysicalDeviceHandle, QueueAssignment>& match) -> uint32_t {
        uint32_t count = 0;
        for (const auto& name : desired_additional_features.get_enabled_features())
            count += static_cast<uint32_t>(match.first->get_supported_features().get_feature(name));
        for (const auto& [_, result] : query_support(match).results)
            for (const auto& feat : result.required_features)
                count +=
                    static_cast<uint32_t>(match.first->get_supported_features().get_feature(feat));
        return count;
    };

    auto best = std::max_element(matches.begin(), matches.end(), [&](const auto& a, const auto& b) {
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

    std::string driver_id;
    const char* driver_info;
    if (physical_device->get_properties().is_available<vk::PhysicalDeviceDriverProperties>()) {
        const vk::PhysicalDeviceDriverProperties& driver_props = physical_device->get_properties();
        driver_id = vk::to_string(driver_props.driverID);
        driver_info = driver_props.driverInfo.data();
    } else {
        driver_id = fmt::format("<{} not supported>", VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME);
        driver_info = driver_id.c_str();
    }

    const vk::PhysicalDeviceProperties& props = physical_device->get_properties();
    SPDLOG_INFO("selected physical device {}, vendor id: {}, device id: {}, driver: {}, {}",
                props.deviceName.data(), props.vendorID, props.deviceID, driver_id, driver_info);

    const auto support = query_support(*best);
    queue_info = std::move(best->second);

    return support;
}

namespace {
// Requests for the same capabilities, collapsed into the family the context has to find.
struct MergedQueueRequest {
    vk::QueueFlags capabilities{};
    bool dedicated = false;
    uint32_t count = 0;
    bool required = false;
    std::vector<std::function<bool(const PhysicalDeviceHandle&, uint32_t)>> prefer_family;
};

uint32_t capability_count(const vk::QueueFlags capabilities) {
    return std::popcount(static_cast<uint32_t>(static_cast<VkQueueFlags>(capabilities)));
}

uint64_t queue_key(const uint32_t family_index, const uint32_t queue_index) {
    return (static_cast<uint64_t>(family_index) << 32U) | queue_index;
}

std::vector<MergedQueueRequest> merge_queue_requests(const std::vector<QueueRequest>& requests) {
    // Widest first, so a request that only needs a subset of another one finds it. Never widen an
    // entry: a combination can be unsatisfiable where each request alone is not.
    std::vector<QueueRequest> widest_first = requests;
    std::stable_sort(widest_first.begin(), widest_first.end(),
                     [](const QueueRequest& a, const QueueRequest& b) {
                         return capability_count(a.capabilities) > capability_count(b.capabilities);
                     });

    std::vector<MergedQueueRequest> merged;
    for (const QueueRequest& request : widest_first) {
        const auto it =
            std::find_if(merged.begin(), merged.end(), [&](const MergedQueueRequest& other) {
                if (other.dedicated != request.dedicated) {
                    return false;
                }
                return request.dedicated
                           ? other.capabilities == request.capabilities
                           : (other.capabilities & request.capabilities) == request.capabilities;
            });
        auto entry = it;
        if (entry == merged.end()) {
            entry = merged.emplace(merged.end());
            entry->capabilities = request.capabilities;
            entry->dedicated = request.dedicated;
        }
        entry->count = std::max(entry->count, request.count);
        entry->required |= request.required;
        if (request.prefer_family) {
            entry->prefer_family.emplace_back(request.prefer_family);
        }
    }
    return merged;
}

} // namespace

std::vector<QueueRequest>
Context::collect_queue_requests(const PhysicalDeviceHandle& physical_device,
                                const std::vector<QueueRequest>& requests) {
    std::vector<QueueRequest> all_requests = requests;
    for (const auto& ext : get_extensions()) {
        insert_all(all_requests, ext->request_queues(physical_device));
    }
    return all_requests;
}

std::optional<QueueAssignment>
Context::determine_queues(const PhysicalDeviceHandle& physical_device,
                          const std::vector<QueueRequest>& requests) const {
    const std::vector<vk::QueueFamilyProperties> families =
        physical_device->get_physical_device().getQueueFamilyProperties();
    if (families.empty()) {
        throw std::runtime_error{"no queue families available!"};
    }

#ifndef NDEBUG
    for (uint32_t family = 0; family < families.size(); family++) {
        SPDLOG_DEBUG("queue family {}: {} queues, {}", family, families[family].queueCount,
                     vk::to_string(families[family].queueFlags));
    }
#endif

    QueueAssignment assignment;
    std::vector<uint32_t> used_per_family(families.size(), 0);

    // Requests are satisfied in order, so the ones listed first get the better family.
    for (const MergedQueueRequest& request : merge_queue_requests(requests)) {
        // (preferences met, family not taken yet, capabilities that were not asked for, free
        // queues, family index), larger is better.
        std::optional<std::tuple<uint32_t, bool, int32_t, uint32_t, int32_t>> best_score;
        std::optional<uint32_t> best_family;

        for (uint32_t family = 0; family < families.size(); family++) {
            const uint32_t free_queues = families[family].queueCount - used_per_family[family];
            if ((families[family].queueFlags & request.capabilities) != request.capabilities ||
                free_queues == 0) {
                continue;
            }

            uint32_t preferences_met = 0;
            for (const auto& prefer : request.prefer_family) {
                preferences_met += static_cast<uint32_t>(prefer(physical_device, family));
            }
            const auto unrelated = static_cast<uint32_t>(
                static_cast<VkQueueFlags>(families[family].queueFlags & ~request.capabilities));
            const std::tuple score{
                preferences_met, used_per_family[family] == 0,  -std::popcount(unrelated),
                free_queues,     -static_cast<int32_t>(family),
            };

            if (!best_score || score > *best_score) {
                best_score = score;
                best_family = family;
            }
        }

        if (!best_family) {
            if (request.required) {
                SPDLOG_DEBUG("no queue family supports the required queue {}",
                             vk::to_string(request.capabilities));
                return std::nullopt;
            }
            continue;
        }

        const uint32_t count = std::min(request.count, families[*best_family].queueCount -
                                                           used_per_family[*best_family]);
        std::vector<uint32_t> queue_indices;
        for (uint32_t i = 0; i < count; i++) {
            queue_indices.emplace_back(used_per_family[*best_family]++);
        }
        SPDLOG_DEBUG("queue {}: family {}, indices [{}]", vk::to_string(request.capabilities),
                     *best_family, fmt::join(queue_indices, ", "));
        assignment.add({request.capabilities, *best_family, std::move(queue_indices)});
    }

    return assignment;
}

Context::FeatureExtensionCheckResult Context::determine_features_extensions(
    const VulkanFeatures& desired_additional_features,
    const std::vector<const char*>& desired_additional_extensions,
    const DeviceSupportCache& support_cache) {

    VulkanFeatures all_desired_features = desired_additional_features;
    std::vector<const char*> all_desired_extensions = desired_additional_extensions;

    const uint32_t effective_device_vk_api_version = physical_device->get_vk_api_version();

    for (const auto& ext : get_extensions()) {
        const auto& support_info = support_cache.results.at(ext);
        insert_all(all_desired_extensions, support_info.required_extensions);
        for (const auto* feature_name : support_info.required_features) {
            all_desired_features.set_feature(feature_name, true);
        }
        for (const auto* spirv_ext : support_info.required_spirv_extensions)
            insert_all(all_desired_extensions, get_spirv_extension_requirements(
                                                   spirv_ext, effective_device_vk_api_version));
        for (const auto* spirv_cap : support_info.required_spirv_capabilities) {
            insert_all(all_desired_extensions,
                       get_spirv_capability_extensions(spirv_cap, effective_device_vk_api_version));
            all_desired_features.enable_features(
                get_spirv_capability_features(spirv_cap, effective_device_vk_api_version));
        }
    }

    Context::FeatureExtensionCheckResult result;

    for (const char* ext : all_desired_features.get_required_extensions()) {
        const ExtensionInfo* ext_info = get_extension_info(ext);
        if (ext_info != nullptr && ext_info->is_instance_extension()) {
            if (instance->get_vk_api_version() < ext_info->promoted_to_version &&
                !instance->extension_enabled(ext) && supported_instance_extensions.contains(ext) &&
                supported_instance_extensions.at(ext).supported) {
                result.missing_instance_extensions.emplace_back(ext);
            }
        } else {
            all_desired_extensions.emplace_back(ext);
        }
    }

    sort_and_remove_duplicates(all_desired_extensions);

    SPDLOG_DEBUG("checking features...");
    for (const auto& feature_name : all_desired_features.get_enabled_features()) {
        if (physical_device->get_supported_features().get_feature(feature_name)) {
            result.features.set_feature(feature_name, true);
            SPDLOG_DEBUG("enable {}", feature_name);
        } else {
            SPDLOG_WARN("{} requested but not supported", feature_name);
        }
    }

    SPDLOG_DEBUG("checking extensions...");
    for (uint32_t i = 0; i < all_desired_extensions.size(); i++) {
        const char* ext = all_desired_extensions[i];
        const auto support_it = physical_device->get_extension_support().find(ext);
        if (support_it == physical_device->get_extension_support().end()) {
            SPDLOG_WARN("removed device extension {} that is not advertised as supported", ext);
            continue;
        }
        if (!support_it->second.supported) {
            SPDLOG_WARN("removed device extension {} that is unsupported because {}", ext,
                        support_it->second.info);
            insert_all(result.missing_instance_extensions,
                       support_it->second.missing_instance_extensions);
            continue;
        }
        const ExtensionInfo* ext_info = get_extension_info(ext);
        if (ext_info != nullptr &&
            ext_info->promoted_to_version <= effective_device_vk_api_version) {
            SPDLOG_DEBUG("removed device extension {} that is already provided by API version",
                         ext);
            continue;
        }
        if (ext_info != nullptr && ext_info->deprecated_by != nullptr &&
            physical_device->extension_supported(ext_info->deprecated_by->name)) {
            SPDLOG_DEBUG("removed deprecated device extension {} and replaced with {}", ext,
                         ext_info->deprecated_by->name);
            continue;
        }

        insert_all(all_desired_extensions, support_it->second.dependencies);

        SPDLOG_DEBUG("enable {}", ext);
        result.extensions.emplace_back(ext);
    }

    sort_and_remove_duplicates(result.extensions);

    return result;
}

void Context::create_device_and_queues(const std::vector<QueueRequest>& requests,
                                       VulkanFeatures& features,
                                       std::vector<const char*>& extensions) {

    // -------------------------------
    // PREPARE QUEUES

    // Re-assign: extensions that turned out to be unsupported were removed since the physical
    // device was selected, and their queues must not be created.
    std::optional<QueueAssignment> assignment =
        determine_queues(physical_device, collect_queue_requests(physical_device, requests));
    if (!assignment) {
        throw MerianException{"the selected device can no longer provide the required queues"};
    }
    queue_info = std::move(*assignment);

    const std::vector<vk::QueueFamilyProperties> families =
        physical_device->get_physical_device().getQueueFamilyProperties();
    std::vector<uint32_t> count_per_family(families.size());
    for (uint32_t family = 0; family < families.size(); family++) {
        count_per_family[family] = queue_info.queues_in_family(family);
    }

    const uint32_t max_queue_count =
        *std::max_element(count_per_family.begin(), count_per_family.end());
    const std::vector<float> queue_priorities(max_queue_count, 1.0f);

    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    for (uint32_t family = 0; family < families.size(); family++) {
        if (count_per_family[family] > 0) {
            queue_create_infos.push_back(
                {{}, family, count_per_family[family], queue_priorities.data()});
        }
    }

    // -------------------------------

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

    // Make discovered plugins' resources (shaders, data) reachable, so plugin nodes can load e.g.
    // "shader/foo.slang" without hardcoding paths. Inherited by the shader compile context.
    file_loader->add_search_path(Plugins::resource_search_paths());
}

///////////////
// GETTER
///////////////

void QueueAssignment::add(const QueueInfo& queue) {
    queues.emplace_back(queue);
}

std::optional<QueueInfo> QueueAssignment::get(const vk::QueueFlags capabilities) const {
    std::optional<QueueInfo> best;
    for (const QueueInfo& queue : queues) {
        if (queue.queue_indices.empty() || (queue.capabilities & capabilities) != capabilities) {
            continue;
        }
        if (queue.capabilities == capabilities) {
            return queue;
        }
        if (!best || capability_count(queue.capabilities) < capability_count(best->capabilities)) {
            best = queue;
        }
    }
    return best;
}

bool QueueAssignment::supports(const vk::QueueFlags capabilities) const {
    return get(capabilities).has_value();
}

uint32_t QueueAssignment::queues_in_family(const uint32_t family_index) const {
    uint32_t count = 0;
    for (const QueueInfo& queue : queues) {
        if (queue.family_index == family_index) {
            count += static_cast<uint32_t>(queue.queue_indices.size());
        }
    }
    return count;
}

std::shared_ptr<Queue> Context::get_queue(const vk::QueueFlags capabilities, const uint32_t index) {
    const std::optional<QueueInfo> info = queue_info.get(capabilities);
    if (!info || index >= info->queue_indices.size()) {
        return nullptr;
    }

    const uint64_t key = queue_key(info->family_index, info->queue_indices[index]);
    if (const auto cached = queues.find(key); cached != queues.end() && !cached->second.expired()) {
        return cached->second.lock();
    }

    const auto queue =
        std::make_shared<Queue>(shared_from_this(), info->family_index, info->queue_indices[index]);
    queues[key] = queue;
    return queue;
}

uint32_t Context::get_queue_count(const vk::QueueFlags capabilities) const {
    const std::optional<QueueInfo> info = queue_info.get(capabilities);
    return info ? static_cast<uint32_t>(info->queue_indices.size()) : 0;
}

std::shared_ptr<CommandPool> Context::get_cmd_pool_GCT() {
    if (!cmd_pool_GCT.expired()) {
        return cmd_pool_GCT.lock();
    }
    const auto cmd = CommandPool::create(get_queue(
        vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer));
    cmd_pool_GCT = cmd;
    return cmd;
}

std::shared_ptr<CommandPool> Context::get_cmd_pool_T() {
    if (!cmd_pool_T.expired()) {
        return cmd_pool_T.lock();
    }
    const auto cmd = CommandPool::create(get_queue(vk::QueueFlagBits::eTransfer));
    cmd_pool_T = cmd;
    return cmd;
}

std::shared_ptr<CommandPool> Context::get_cmd_pool_C() {
    if (!cmd_pool_C.expired()) {
        return cmd_pool_C.lock();
    }
    const auto cmd = CommandPool::create(get_queue(vk::QueueFlagBits::eCompute));
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

const QueueAssignment& Context::get_queue_info() const {
    return queue_info;
}

const ShaderCompileContextHandle& Context::get_shader_compile_context() const {
    return shader_compile_context;
}

} // namespace merian
