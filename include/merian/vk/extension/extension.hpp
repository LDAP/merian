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
    /* Extensions that should be enabled device-wide. Note that on_physical_device_selected is
     * called before. */
    virtual std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const {
        return {};
    }

    // LIFECYCLE (in order)

    /**
     * Append structs to vkInstanceCreateInfo to enable features of extensions.
     *
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* pnext_instance_create_info(void* const p_next) {
        return p_next;
    }
    virtual void on_instance_created(const vk::Instance&) {}
    /* Called after the physical device was select and before extensions are checked for
     * compativility and check_support is called.*/
    virtual void on_physical_device_selected(const Context::PhysicalDeviceContainer&) {}

    /* Append a structure to pNext of a getFeatures() call. This can be used to determine extension
     * support.
     *
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* pnext_get_features_2(void* const p_next) {
        return p_next;
    }

    /* Custom check for compatibility after the physical device is ready.
     * If this method returns false, it is guaranteed that on_unsupported is called.
     */
    virtual bool extension_supported(const Context::PhysicalDeviceContainer&) {
        return true;
    }
    /* E.g. to dismiss a queue that does not support present-to-surface. */
    virtual bool accept_graphics_queue([[maybe_unused]] const vk::Instance& instance,
                                       [[maybe_unused]] const vk::PhysicalDevice& physical_device,
                                       [[maybe_unused]] std::size_t queue_family_index) {
        return true;
    }
    /**
     * Append structs to VkDeviceCreateInfo to enable features of extensions.
     *
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* pnext_device_create_info(void* const p_next) {
        return p_next;
    }
    /* Do not change pNext. You can use pnext_device_create_info for that. */
    virtual void enable_device_features(const Context::FeaturesContainer&,
                                        Context::FeaturesContainer&) {}
    virtual void on_device_created(const vk::Device&) {}
    /* Called right before context constructor returns. */
    virtual void on_context_created(const ContextHandle) {}
    /* Called after device is idle and before context is destroyed. */
    virtual void on_destroy_context() {}
    /* Called right before device is destroyed. */
    virtual void on_destroy_device(const vk::Device&) {}
    /* Called before the instance is destroyed or if the extension is determined as unsupported. */
    virtual void on_destroy_instance(const vk::Instance&) {}
    // Called by context if extension was determined as unsupported. The extension might not receive
    // further callbacks.
    virtual void on_unsupported([[maybe_unused]] const std::string reason) {
        spdlog::warn("extension {} not supported ({})", name, reason);
    }

  public:
    const std::string name;
};

} // namespace merian
