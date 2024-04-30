#pragma once

#include <spdlog/logger.h>

#include <vulkan/vulkan.hpp>

namespace merian {

// cyclic -> forward definition
class Extension;
class Context;
class Queue;
class CommandPool;

using SharedContext = std::shared_ptr<Context>;

/* Initializes the Vulkan instance and device and holds core objects.
 *
 * Common features are automatically enabled.
 *
 * Extensions can extend the functionality and hook into the creation process.
 * Use SharedContext instead of Context directly. This way it is ensured that Context is destroyed
 * last.
 */
class Context : public std::enable_shared_from_this<Context> {

  public:
    struct FeaturesContainer {
        operator const vk::PhysicalDeviceFeatures2&() const {
            return physical_device_features;
        }

        operator vk::PhysicalDeviceFeatures2&() {
            return physical_device_features;
        }

        operator const vk::PhysicalDeviceFeatures&() const {
            return physical_device_features.features;
        }

        operator vk::PhysicalDeviceFeatures&() {
            return physical_device_features.features;
        }

        vk::PhysicalDeviceFeatures2 physical_device_features;
        vk::PhysicalDeviceVulkan11Features physical_device_features_v11;
        vk::PhysicalDeviceVulkan12Features physical_device_features_v12;
        vk::PhysicalDeviceVulkan13Features physical_device_features_v13;
    };

    struct PhysicalDeviceContainer {
        operator const vk::PhysicalDevice&() const {
            return physical_device;
        }

        operator vk::PhysicalDevice&() {
            return physical_device;
        }

        const vk::PhysicalDeviceLimits& get_physical_device_limits() const {
            return physical_device_properties.properties.limits;
        }

        vk::PhysicalDevice physical_device;
        vk::PhysicalDeviceProperties2 physical_device_properties;
        vk::PhysicalDeviceMemoryProperties2 physical_device_memory_properties;
        vk::PhysicalDeviceSubgroupProperties physical_device_subgroup_properties;
        std::vector<vk::ExtensionProperties> physical_device_extension_properties;
        FeaturesContainer features;
    };

    /**
     * @brief      Use this method to create the context.
     *
     *
     * Needed for enable_shared_from_this, see
     * https://en.cppreference.com/w/cpp/memory/enable_shared_from_this.
     */
    static SharedContext
    make_context(std::vector<std::shared_ptr<Extension>> extensions,
                 std::string application_name = "",
                 uint32_t application_vk_version = VK_MAKE_VERSION(1, 0, 0),
                 uint32_t preffered_number_compute_queues = 1, // Additionally to the GCT queue
                 uint32_t filter_vendor_id = -1,
                 uint32_t filter_device_id = -1,
                 std::string filter_device_name = "");

  private:
    Context(std::vector<std::shared_ptr<Extension>> extensions,
            std::string application_name,
            uint32_t application_vk_version,
            uint32_t preffered_number_compute_queues,
            uint32_t filter_vendor_id,
            uint32_t filter_device_id,
            std::string filter_device_name);

  public:
    ~Context();

    operator vk::Instance&() {
        return instance;
    }

    operator vk::PhysicalDevice&() {
        return physical_device;
    }

    operator vk::Device&() {
        return device;
    }

  private: // Vulkan initialization
    void create_instance(std::string application_name, uint32_t application_vk_version);
    void prepare_physical_device(uint32_t filter_vendor_id,
                                 uint32_t filter_device_id,
                                 std::string filter_device_name);
    void find_queues();
    void create_device_and_queues(uint32_t preffered_number_compute_queues);

  private: // Helper
    void extensions_check_instance_layer_support();
    void extensions_check_instance_extension_support();
    void extensions_check_device_extension_support();
    void extensions_self_check_support();
    void destroy_extensions(std::vector<std::shared_ptr<Extension>> extensions);

  public: // Getter
    // can be nullptr in very rare occasions, you can check that with if (shrd_ptr) {...}
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<Queue> get_queue_GCT();

    // can be nullptr in very rare occasions, you can check that with if (shrd_ptr) {...}
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<Queue> get_queue_T();

    // can be nullptr in very rare occasions, you can check that with if (shrd_ptr) {...}
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<Queue> get_queue_C(uint32_t index = 0);

    // Convenience command pool for graphics and compute (can be nullptr in very rare occasions)
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<CommandPool> get_cmd_pool_GCT();

    // Convenience command pool for transfer (can be nullptr in very rare occasions)
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<CommandPool> get_cmd_pool_T();

    // Convenience command pool for compute (can be nullptr in very rare occasions)
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<CommandPool> get_cmd_pool_C();

    template <class Extension> std::shared_ptr<Extension> get_extension() {
        for (auto& ext : extensions) {
            if (std::shared_ptr<Extension> casted_extension =
                    dynamic_pointer_cast<Extension>(ext)) {
                return casted_extension;
            }
        }
        return nullptr;
    }

    bool device_extension_enabled(const std::string& name);

    bool instance_extension_enabled(const std::string& name);

  private:
    std::vector<std::shared_ptr<Extension>> extensions;

    // in create_instance

    std::vector<const char*> instance_layer_names;
    std::vector<const char*> instance_extension_names;

  public:
    std::string application_name;
    uint32_t vk_api_version = VK_API_VERSION_1_3;
    uint32_t application_vk_version;

    // in create_instance

    vk::Instance instance;

    // in prepare_physical_device

    // the vk::PhysicalDevice for this Context
    PhysicalDeviceContainer physical_device;

    // in find_queues. Indexes are -1 if no suitable queue was found!

    // A queue family index guaranteed to support graphics+compute+graphics
    int32_t queue_family_idx_GCT = -1;
    // A queue family index guaranteed to support compute
    int32_t queue_family_idx_C = -1;
    // A queue family index guaranteed to support transfer
    int32_t queue_family_idx_T = -1;

    // in create_device_and_queues

    std::vector<const char*> device_extensions;
    // the vk::Device for this Context
    vk::Device device;
    // The queue indices (not family!) used for the graphics queue
    int32_t queue_idx_GCT = -1;
    // The queue indices (not family!) used for the compute queue
    std::vector<uint32_t> queue_idx_C;
    // The queue indices (not family!) used for the transfer queue
    int32_t queue_idx_T = -1;

  private:
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
};

} // namespace merian

#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/command/queue.hpp"
