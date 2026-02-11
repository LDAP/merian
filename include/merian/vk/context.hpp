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
    // returns the context extension if loaded, otherwise nullptr (if null_ok == true) or throws
    // MissingExtension.
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
    // returns the extensions ordered by their requirements
    const std::vector<std::shared_ptr<ContextExtension>>& get_extensions() const {
        return ordered_extensions;
    }

    void add_extension(const std::shared_ptr<ContextExtension>& extension);

    void remove_extension(const std::type_index& type);

  private:
    std::unordered_map<std::type_index, std::shared_ptr<ContextExtension>> context_extensions;
    std::vector<std::shared_ptr<ContextExtension>> ordered_extensions;
};

using ConfigureExtensionsCallback = std::function<void(ExtensionContainer&)>;

struct ContextCreateInfo {
    VulkanFeatures desired_features{};
    std::vector<const char*> additional_extensions{};
    std::vector<std::string> context_extensions{};
    std::optional<ConfigureExtensionsCallback> configure_extensions_callback{};
    std::vector<std::filesystem::path> additional_search_paths{};
    std::string application_name = "";
    uint32_t application_vk_version = VK_MAKE_VERSION(1, 0, 0);
    uint32_t preferred_number_compute_queues = 1;
    uint32_t filter_vendor_id = -1;
    uint32_t filter_device_id = -1;
    std::string filter_device_name = "";
};

struct QueueInfo {
    // A queue family index guaranteed to support graphics+compute+transfer (or -1)
    int32_t queue_family_idx_GCT = -1;
    // A queue family index guaranteed to support compute (or -1)
    int32_t queue_family_idx_C = -1;
    // A queue family index guaranteed to support transfer (or -1)
    int32_t queue_family_idx_T = -1;

    // The queue indices (not family!) used for the graphics queue
    int32_t queue_idx_GCT = -1;
    // The queue indices (not family!) used for the compute queue
    std::vector<uint32_t> queue_idx_C;
    // The queue indices (not family!) used for the transfer queue
    int32_t queue_idx_T = -1;
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
    void create_instance(const uint32_t targeted_vk_api_version,
                         const VulkanFeatures& desired_features,
                         const std::vector<const char*>& desired_additional_extensions);
    void select_physical_device(uint32_t filter_vendor_id,
                                uint32_t filter_device_id,
                                std::string filter_device_name,
                                const VulkanFeatures& desired_features,
                                const std::vector<const char*>& desired_additional_extensions);
    QueueInfo determine_queues(const PhysicalDeviceHandle& physical_device);
    void create_device_and_queues(uint32_t preffered_number_compute_queues,
                                  const VulkanFeatures& desired_features,
                                  const std::vector<const char*>& desired_additional_extensions);
    void prepare_file_loader(const ContextCreateInfo& create_info);

  public: // Getter
    // The actual number of compute queues (< preffered_number_compute_queues).
    uint32_t get_number_compute_queues() const noexcept;

    // A queue guaranteed to support graphics+compute+transfer.
    // Can be nullptr if unavailable, you can check that with if (shrd_ptr) {...}
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<Queue> get_queue_GCT();

    // A queue guaranteed to support transfer.
    // Can be nullptr if unavailable, you can check that with if (shrd_ptr) {...}
    // Make sure to keep a reference, else the pool and its buffers are destroyed.
    // Might fall back to the GCT queue if fallback is true.
    std::shared_ptr<Queue> get_queue_T(const bool fallback = false);

    // A queue guaranteed to support compute.
    // Can be nullptr if unavailable, you can check that with if (shrd_ptr) {...}
    // Make sure to keep a reference, else the pool and its buffers are destroyed.
    // Might fall back to a different compute queue or the GCT queue, if fallback is true.
    // In this case index may be invalid, else it must be < get_number_compute_queues() and or
    // nullptr is returned.
    std::shared_ptr<Queue> get_queue_C(uint32_t index = 0, const bool fallback = false);

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

    const QueueInfo& get_queue_info() const;

    const ShaderCompileContextHandle& get_shader_compile_context() const;

  private:
    // in create_device_and_queues

    const std::string application_name;
    const uint32_t application_vk_version;

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

    // in find_queues. Indexes are -1 if no suitable queue was found!

    QueueInfo queue_info;

    // can be nullptr in very rare occasions, you can check that with if (shrd_ptr) {...}
    std::weak_ptr<Queue> queue_GCT;
    // can be nullptr in very rare occasions, you can check that with if (shrd_ptr) {...}
    std::weak_ptr<Queue> queue_T;
    // resized in create_device_and_queues
    std::vector<std::weak_ptr<Queue>> queues_C;

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
