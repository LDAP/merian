#pragma once

#include "merian/vk/command/queue_container.hpp"

#include <mutex>
#include <spdlog/logger.h>

#include <vulkan/vulkan.hpp>

namespace merian {

// cyclic -> forward definition
class Extension;

/* Initializes the Vulkan instance and device and holds core objects.
 *
 * Extensions can extend the functionality and hook into the creation process.
 */
class Context {
  public:
    Context(std::vector<Extension*> extensions, std::string application_name = "",
            uint32_t application_vk_version = VK_MAKE_VERSION(1, 0, 0), uint32_t filter_vendor_id = -1,
            uint32_t filter_device_id = -1, std::string filter_device_name = "");
    ~Context();

  private: // Vulkan initialization
    void create_instance(std::string application_name, uint32_t application_vk_version);
    void prepare_physical_device(uint32_t filter_vendor_id, uint32_t filter_device_id, std::string filter_device_name);
    void find_queues();
    void create_device_and_queues();
    void create_command_pools();

  private: // Helper
    void extensions_check_instance_layer_support();
    void extensions_check_instance_extension_support();
    void extensions_check_device_extension_support();
    void extensions_self_check_support();
    void destroy_extensions(std::vector<Extension*> extensions);

  private:
    std::vector<Extension*> extensions;

    // in create_instance

    std::vector<const char*> instance_layer_names;
    std::vector<const char*> instance_extension_names;

  public:
    std::string application_name;
    uint32_t application_vk_version;

    // in create_instance

    vk::Instance instance;

    // in prepare_physical_device

    // the vk::PhysicalDevice for this Context
    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties2 physical_device_props;
    vk::PhysicalDeviceFeatures2 physical_device_features;
    vk::PhysicalDeviceMemoryProperties2 physical_device_memory_properties;
    std::vector<vk::ExtensionProperties> physical_device_extension_properties;

    // in find_queues TODO: Separate compute, provide multiple queues

    // A queue family index guaranteed to support graphics
    uint32_t queue_idx_graphics = -1;
    // A queue family index guaranteed to support transfer
    uint32_t queue_idx_transfer = -1;

    // in create_device_and_queues

    // the vk::Device for this Context
    vk::Device device;
    // can be used for both graphics and compute
    QueueContainer* queue_graphics;
    QueueContainer* queue_transfer;

    // in create_command_pools

    // Convenience command pool for graphics and compute
    vk::CommandPool cmd_pool_graphics;
    // Convenience command pool for transfer
    vk::CommandPool cmd_pool_transfer;
};

} // namespace merian
