#pragma once
#include <mutex>
#include <spdlog/logger.h>

#define VK_ENABLE_BETA_EXTENSIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
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

  public:
    std::string application_name;
    uint32_t application_vk_version;
    std::shared_ptr<spdlog::logger> logger;
    // in create_instance
    std::vector<const char*> instance_layer_names;
    std::vector<const char*> instance_extension_names;
    vk::Instance instance;
    // in prepare_physical_device
    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties2 physical_device_props;
    vk::PhysicalDeviceFeatures2 physical_device_features;
    vk::PhysicalDeviceMemoryProperties2 physical_device_memory_properties;
    std::vector<vk::ExtensionProperties> physical_device_extension_properties;
    // in find_queues
    uint32_t queue_idx_graphics = -1;
    uint32_t queue_idx_transfer = -1;
    // in create_device_and_queues
    vk::Device device;
    vk::Queue queue_graphics; // used for both graphics and compute
    vk::Queue queue_transfer;
    std::mutex queue_graphics_mutex;
    std::mutex queue_transfer_mutex;
    // in create_command_pools
    vk::CommandPool cmd_pool_graphics;
    vk::CommandPool cmd_pool_transfer;
};

} // namespace merian
