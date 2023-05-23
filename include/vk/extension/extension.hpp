#pragma once
#define VK_ENABLE_BETA_EXTENSIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vector>
#include <vulkan/vulkan.hpp>

// cyclic -> forward definition
class Context;

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
     * Append structs to vkInstanceCreateInfo.
     *
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* on_create_instance(void* p_next) {
        return p_next;
    }
    virtual void on_instance_created(vk::Instance&) {}
    virtual void on_destroy_instance(vk::Instance&) {}
    virtual bool accept_graphics_queue(vk::PhysicalDevice&, std::size_t) {
        return true;
    }
    /**
     * Append structs to VkDeviceCreateInfo.
     *
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* on_create_device(void* p_next) {
        return p_next;
    }
    virtual void on_device_created(vk::Device&) {}
    virtual void on_context_created(Context&) {}
    virtual void on_destroy_context(Context&) {}
    virtual bool extension_supported(vk::PhysicalDevice&, std::vector<vk::ExtensionProperties>& extension_properties) {
        bool supported = true;
        for (const char* required_extension : required_device_extension_names()) {
            bool extension_found = false;
            for (auto& props : extension_properties) {
                if (!strcmp(props.extensionName, required_extension)) {
                    extension_found = true;
                }
            }
            if (!extension_found) {
                supported = false;
            }
        }
        return supported;
    }
};
