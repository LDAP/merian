#pragma once
#include "vk/extension/extension.hpp"
#include <mutex>
#include <vulkan/vulkan.hpp>

class Context {
  public:
    Context(uint32_t vendor_id = -1, uint32_t device_id = -1);
    ~Context();

  private:
    void create_instance();
    void prepare_physical_device(uint32_t vendor_id = -1, uint32_t device_id = -1);
    void find_queues();
    void create_device_and_queues();
    void create_command_pools();

  public:
    std::vector<Extension*> extensions;
    // in create_instance
    std::vector<const char*> layer_names;
    std::vector<const char*> instance_extension_names;
    vk::Instance instance;
    // in prepare_physical_device
    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties2 physical_device_props;
    vk::PhysicalDeviceFeatures2 physical_device_features;
    vk::PhysicalDeviceMemoryProperties2 physical_device_memory_properties;
    std::vector<vk::ExtensionProperties> extension_properties;
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
