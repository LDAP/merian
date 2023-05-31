#pragma once

#include "merian/vk/context.hpp"
#include <spdlog/spdlog.h>
#include <vector>

namespace merian {

/**
 * @brief      An extension to the Vulkan Context.
 *
 * An extension can enable layers and Vulkan instance and device extensions (_EXT), as well as
 * hook into the context creation process. Extensions are checked for compatibility and the result
 * can be retrived using is_supported().
 *
 * If a extension is determined to be not compatible it is removed from the context and
 * the corresponding on_destroy_* lifecycle methods are called.
 */
class Extension {

    friend Context;

  public:
    Extension(std::string name) : name(name) {}

    virtual ~Extension() = 0;

    // REQUIREMENTS

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

    // LIFECYCLE (in order)

    /**
     * Append structs to vkInstanceCreateInfo.
     *
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* on_create_instance(void* const p_next) {
        return p_next;
    }
    virtual void on_instance_created(const vk::Instance&) {}
    /* Called after the physical device was select and before extensions are checked for compativility and check_support
     * is called. */
    virtual void on_physical_device_selected(const vk::PhysicalDevice&) {}
    /* Custom check for compatibility after the physical device is ready. */
    virtual bool extension_supported(const vk::PhysicalDevice&) {
        return true;
    }
    /* E.g. to dismiss a queue that does not support present-to-surface. */
    virtual bool accept_graphics_queue(const vk::PhysicalDevice&, std::size_t) {
        return true;
    }
    /**
     * Append structs to VkDeviceCreateInfo.
     *
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* on_create_device(void* const p_next) {
        return p_next;
    }
    virtual void on_device_created(const vk::Device&) {}
    /* Called right before context constructor returns. */
    virtual void on_context_created(const Context&) {}
    /* Called after device is idle and before context is destroyed. */
    virtual void on_destroy_context(const Context&) {}
    /* Called right before device is destroyed. */
    virtual void on_destroy_device(const vk::Device&) {}
    /* Called before the instance is destroyed or if the extension is determined as unsupported. */
    virtual void on_destroy_instance(const vk::Instance&) {}

    // OTHER

    /* Only valid after context initialization */
    virtual bool is_supported() final {
        return supported;
    }

  public:
    const std::string name;

  private:
    // written by Context
    bool supported = true;
};

} // namespace merian
