#pragma once

#include <spdlog/logger.h>

#include <typeindex>

#include "merian/vk/device.hpp"
#include "merian/vk/instance.hpp"
#include "merian/vk/physical_device.hpp"

#include "merian/fwd.hpp"
#include "merian/io/file_loader.hpp"
#include "merian/vk/utils/vulkan_features.hpp"

namespace merian {

class MerianException : public std::runtime_error {
  public:
    MerianException(const std::string& reason) : std::runtime_error(reason) {}
};

class MissingExtension : public MerianException {
  public:
    MissingExtension(const std::string& reason) : MerianException(reason) {}
};

class VulkanException : public MerianException {
  public:
    VulkanException(const vk::Result result)
        : MerianException(fmt::format("call failed with {}", vk::to_string(result))),
          result(result) {}

    VulkanException(const VkResult result) : merian::VulkanException(vk::Result(result)) {}

    VulkanException(const vk::Result result, const std::string& additional_info)
        : MerianException(
              fmt::format("call failed with {}. {}", vk::to_string(result), additional_info)),
          result(result) {}

    VulkanException(const VkResult result, const std::string& additional_info)
        : merian::VulkanException(vk::Result(result), additional_info) {}

    const vk::Result& get_result() const {
        return result;
    }

    static void throw_if_no_success(const vk::Result result) {
        if (result != vk::Result::eSuccess) {
            throw VulkanException(result);
        }
    }

    static void throw_if_no_success(const VkResult result) {
        if (result != VK_SUCCESS) {
            throw VulkanException(result);
        }
    }

    static void throw_if_no_success(const vk::Result result, const std::string& additional_info) {
        if (result != vk::Result::eSuccess) {
            throw VulkanException(result, additional_info);
        }
    }

    static void throw_if_no_success(const VkResult result, const std::string& additional_info) {
        if (result != VK_SUCCESS) {
            throw VulkanException(result, additional_info);
        }
    }

  private:
    vk::Result result;
};

class ExtensionContainer {

  public:
    // Returns all loaded extensions that implement or derive from T, sorted by priority (highest
    // first). Priority is looked up per-interface from the ExtensionRegistry.
    template <class T> std::vector<std::shared_ptr<T>> find_providers() const {
        const std::type_index iface = typeid(T);
        std::vector<std::pair<int, std::shared_ptr<T>>> with_priority;
        for (const auto& ext : ordered_extensions) {
            if (auto cast = std::dynamic_pointer_cast<T>(ext)) {
                with_priority.emplace_back(get_extension_priority(ext, iface), std::move(cast));
            }
        }
        std::sort(with_priority.begin(), with_priority.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::shared_ptr<T>> result;
        result.reserve(with_priority.size());
        for (auto& [_, p] : with_priority)
            result.push_back(std::move(p));
        return result;
    }

    // Returns the highest-priority loaded extension that implements or derives from T.
    // Priority is per-interface, looked up from the ExtensionRegistry.
    template <class T> std::shared_ptr<T> find_provider(const bool null_ok = false) const {
        const std::type_index iface = typeid(T);
        std::shared_ptr<T> best;
        int best_priority = std::numeric_limits<int>::min();
        for (const auto& ext : ordered_extensions) {
            if (auto cast = std::dynamic_pointer_cast<T>(ext)) {
                const int p = get_extension_priority(ext, iface);
                if (p > best_priority) {
                    best = std::move(cast);
                    best_priority = p;
                }
            }
        }
        if (best || null_ok) {
            return best;
        }
        throw MissingExtension{fmt::format("no provider for type {} loaded.", typeid(T).name())};
    }

    // Returns the loaded extension of type T, or nullptr / throws MissingExtension.
    template <class ContextExtension>
    std::shared_ptr<ContextExtension> get_context_extension(const bool null_ok = false) const {
        if (context_extensions.contains(typeid(ContextExtension))) {
            return std::static_pointer_cast<ContextExtension>(
                context_extensions.at(typeid(ContextExtension)));
        }
        if (null_ok) {
            return nullptr;
        }
        throw MissingExtension{fmt::format("context extension with type {} not loaded.",
                                           typeid(ContextExtension).name())};
    }

  protected:
    const std::vector<std::shared_ptr<ContextExtension>>& get_extensions() const {
        return ordered_extensions;
    }

    void add_extension(const std::shared_ptr<ContextExtension>& extension);

    void remove_extension(const std::type_index& type);

    // Looks up the per-interface priority for an extension via the ExtensionRegistry.
    // Implemented in context.cpp to avoid a circular include with extension_registry.hpp.
    int get_extension_priority(const std::shared_ptr<ContextExtension>& ext,
                               const std::type_index& interface_type) const;

  private:
    std::unordered_map<std::type_index, std::shared_ptr<ContextExtension>> context_extensions;
    std::vector<std::shared_ptr<ContextExtension>> ordered_extensions;
};

using ConfigureExtensionsCallback = std::function<void(ExtensionContainer&)>;

// A queue the context should create. Requests share a queue whenever one of them already provides
// what the other needs.
struct QueueRequest {
    vk::QueueFlags capabilities;
    // Create a queue of its own instead of sharing one that already provides the capabilities.
    bool dedicated = false;
    // Number of queues to create. Fewer are created if the family is smaller.
    uint32_t count = 1;
    // A device that cannot satisfy this request is not considered.
    bool required = true;
    // Preferred, not required. For support that is not a Vulkan queue capability, such as the
    // presentation support of a window backend.
    std::function<bool(const PhysicalDeviceHandle& physical_device, uint32_t family_index)>
        prefer_family{};
};

struct QueueInfo {
    // The capabilities the queue was requested with. The family may support more.
    vk::QueueFlags capabilities;
    uint32_t family_index;
    // Indices within the family, not family indices.
    std::vector<uint32_t> queue_indices;
};

// The queues the context created, looked up by the capabilities the caller needs.
class QueueAssignment {
  public:
    void add(const QueueInfo& queue);

    // Exactly `capabilities` if such a queue was created, else the closest that provides them.
    std::optional<QueueInfo> get(const vk::QueueFlags capabilities) const;

    bool supports(const vk::QueueFlags capabilities) const;

    uint32_t queues_in_family(const uint32_t family_index) const;

  private:
    std::vector<QueueInfo> queues;
};

struct ContextCreateInfo {
    // Set your desired Vulkan features.
    VulkanFeatures features{};
    // Set your desired instance extensions, extension dependencies are added
    // automatically.
    std::vector<const char*> instance_extensions{};
    // Set your desired device extensions, extension dependencies are added
    // automatically.
    std::vector<const char*> device_extensions{};
    // Set your desired instance layers.
    std::vector<const char*> instance_layers{};
    // Extensions to load by name (from the registry).
    std::vector<std::string> context_extensions{};
    // Called when all extensions are loaded. You can get_context_extension<>() to configure.
    std::optional<ConfigureExtensionsCallback> configure_extensions_callback{};
    // Search paths for shader compiler and file loader.
    std::vector<std::filesystem::path> additional_search_paths{};
    // Your application name.
    std::string application_name = "";
    // Your application version.
    uint32_t application_vk_version = VK_MAKE_VERSION(1, 0, 0);
    // Queues the context creates.
    std::vector<QueueRequest> queues = {
        {.capabilities = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute |
                         vk::QueueFlagBits::eTransfer},
        {.capabilities = vk::QueueFlagBits::eTransfer, .dedicated = true, .required = false},
        {.capabilities = vk::QueueFlagBits::eCompute, .dedicated = true, .required = false},
    };
    // Set to a number greater or equal to 0 to only consider devices with this vendor id.
    uint32_t filter_vendor_id = (uint32_t)-1;
    // Set to a number greater or equal to 0 to only consider devices with this device id.
    uint32_t filter_device_id = (uint32_t)-1;
    // Set to a non-empty string to only consider devices with this name.
    std::string filter_device_name = "";
};

/* Initializes the Vulkan instance and device and holds core objects.
 *
 * Extensions can extend the functionality and hook into the creation process.
 */
class Context : public std::enable_shared_from_this<Context>, public ExtensionContainer {

  public:
#ifdef NDEBUG
    static constexpr bool IS_DEBUG_BUILD = false;
#else
    static constexpr bool IS_DEBUG_BUILD = true;
#endif

#ifdef MERIAN_BUILD_OPTIMIZATION
#if MERIAN_BUILD_OPTIMIZATION == 0
    static constexpr uint32_t BUILD_OPTIMIZATION_LEVEL = 0;
#elif MERIAN_BUILD_OPTIMIZATION == 1
    static constexpr uint32_t BUILD_OPTIMIZATION_LEVEL = 1;
#elif MERIAN_BUILD_OPTIMIZATION == 2
    static constexpr uint32_t BUILD_OPTIMIZATION_LEVEL = 2;
#elif MERIAN_BUILD_OPTIMIZATION == 3
    static constexpr uint32_t BUILD_OPTIMIZATION_LEVEL = 3;
#else
    static constexpr uint32_t BUILD_OPTIMIZATION_LEVEL = 1;
#endif
#else
    static constexpr uint32_t BUILD_OPTIMIZATION_LEVEL = 1;
#endif

    static ContextHandle create(const ContextCreateInfo& create_info);

  private:
    Context(const ContextCreateInfo& create_info);

  public:
    ~Context();

  private: // Vulkan initialization
    void load_extensions(const std::vector<std::string>& extension_names);
    void determine_instance_extension_layer_support(const uint32_t targeted_vk_api_version);
    void create_instance(const uint32_t targeted_vk_api_version,
                         const std::vector<const char*>& desired_additional_extensions,
                         const std::vector<const char*>& desired_additional_layers);
    struct DeviceSupportCache;
    DeviceSupportCache
    select_physical_device(uint32_t filter_vendor_id,
                           uint32_t filter_device_id,
                           std::string filter_device_name,
                           const VulkanFeatures& desired_additional_features,
                           const std::vector<const char*>& desired_additional_extensions,
                           const std::vector<QueueRequest>& queue_requests);
    struct FeatureExtensionCheckResult;
    FeatureExtensionCheckResult
    determine_features_extensions(const VulkanFeatures& desired_additional_features,
                                  const std::vector<const char*>& desired_additional_extensions,
                                  const DeviceSupportCache& support_cache);
    std::vector<QueueRequest> collect_queue_requests(const PhysicalDeviceHandle& physical_device,
                                                     const std::vector<QueueRequest>& requests);

    // nullopt if the device cannot satisfy a required request.
    std::optional<QueueAssignment>
    determine_queues(const PhysicalDeviceHandle& physical_device,
                     const std::vector<QueueRequest>& requests) const;

    void create_device_and_queues(const std::vector<QueueRequest>& requests,
                                  VulkanFeatures& features,
                                  std::vector<const char*>& extensions);
    void prepare_file_loader(const ContextCreateInfo& create_info);

  public: // Getter
    // nullptr if no queue provides `capabilities`.
    // Make sure to keep a reference, else the pool and its buffers are destroyed.
    std::shared_ptr<Queue> get_queue(const vk::QueueFlags capabilities, const uint32_t index = 0);

    uint32_t get_queue_count(const vk::QueueFlags capabilities) const;

    // Convenience command pool for graphics and compute (can be nullptr in very rare occasions)
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<CommandPool> get_cmd_pool_GCT();

    // Convenience command pool for transfer (can be nullptr in very rare occasions)
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<CommandPool> get_cmd_pool_T();

    // Convenience command pool for compute (can be nullptr in very rare occasions)
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<CommandPool> get_cmd_pool_C();

    const InstanceHandle& get_instance() const;

    const PhysicalDeviceHandle& get_physical_device() const;

    const DeviceHandle& get_device() const;

    const FileLoaderHandle& get_file_loader() const;

    const QueueAssignment& get_queue_info() const;

    const ShaderCompileContextHandle& get_shader_compile_context() const;

  private:
    const std::string application_name;
    const uint32_t application_vk_version;

    // in determine_instance_extension_layer_support

    struct InstanceExtensionSupport {
        bool supported;
        std::string info{};
        std::unordered_set<const char*> dependencies{};
    };
    std::unordered_set<std::string> supported_instance_layers;
    std::unordered_map<std::string, InstanceExtensionSupport> supported_instance_extensions;

    // in create_instance

    InstanceHandle instance;

    // in prepare_physical_device

    // the vk::PhysicalDevice for this Context
    PhysicalDeviceHandle physical_device;

    // in create_device_and_queues

    // the vk::Device for this Context
    DeviceHandle device;

    // -----------------

    // A shared file_loader for convenience.
    FileLoaderHandle file_loader;

    // The queues that were created. Requests that could not be satisfied are absent.
    QueueAssignment queue_info;

    // Keyed by family and queue index, so requests resolving to the same queue share its mutex.
    std::unordered_map<uint64_t, std::weak_ptr<Queue>> queues;

    // in make_context

    // Convenience command pool for graphics and compute (can be nullptr in very rare occasions)
    std::weak_ptr<CommandPool> cmd_pool_GCT;
    // Convenience command pool for transfer (can be nullptr in very rare occasions)
    std::weak_ptr<CommandPool> cmd_pool_T;
    // Convenience command pool for compute (can be nullptr in very rare occasions)
    std::weak_ptr<CommandPool> cmd_pool_C;

    ShaderCompileContextHandle shader_compile_context;
};

} // namespace merian

#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/command/queue.hpp"
