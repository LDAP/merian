#pragma once
#define VK_ENABLE_BETA_EXTENSIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vector>

class Extension {
  public:
    virtual ~Extension() = 0;
    virtual std::string name() const = 0;
    virtual std::vector<const char*> required_instance_extension_names() const {
        return {};
    }
    virtual std::vector<const char*> required_layer_names() const {
        return {};
    }
    virtual std::vector<const char*> required_device_extension_names() const {
        return {};
    }
    /**
     * Append structs to InstanceCreateInfo.
     * 
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* on_create_instance(void* p_next) {
      return p_next;
    }
    virtual void on_instance_created(vk::Instance&) {}
    virtual void on_destroy(vk::Instance&) {}
    virtual bool accept_graphics_queue(vk::PhysicalDevice&, std::size_t) { return true; }
};
