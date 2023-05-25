#pragma once

#include "vk/context.hpp"
#include <vector>


/**
 * @brief      An extension to the Vulkan Context.
 *
 * An extension can enable layers and Vulkan instance and device extensions (_EXT), as well as
 * hook into the context creation process. Extensions are checked for compatibility and the result
 * can be retrived using is_supported().
 */
class Extension {

    friend Context;

  public:
    virtual ~Extension() = 0;
    virtual std::string name() const = 0;
    /* Extensions that should be enabled instance-wide. */
    virtual std::vector<const char*> required_instance_extension_names() const {
        return {};
    }
    /* Layers that should be enabled instance-wide. */
    virtual std::vector<const char*> required_instance_layer_names() const {
        return {};
    }
    /* Extensions that should be enabled device-wide. */
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
    /* Called when before the instance is destroyed or if the extension is determined as unsupported */
    virtual void on_destroy_instance(vk::Instance&) {}
    /* E.g. to dismiss a queue that does not support present-to-surface. */
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
    /* Called after device is idle and before context is destroyed. */
    virtual void on_destroy_context(Context&) {}
    /* Custom check for compatibility after the physical device is ready. */
    virtual bool extension_supported(vk::PhysicalDevice&) {
        return true;
    }
    /* Only valid after context initialization */
    virtual bool is_supported() final {
        return supported;
    }

  private:
    // written by Context
    bool supported = true;
};
