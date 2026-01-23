#pragma once

#include "merian/vk/context.hpp"
#include <map>
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
    Extension(const std::string& name) : name(name) {}

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
    virtual std::vector<const char*>
    required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const {
        return {};
    }

    // LIFECYCLE (in order)

    /**
     * Notifies the extensions that the context is starting and allows extensions to communicate.
     *
     * Note, this are the desired extensions and support is yet to be determined.
     */
    virtual void
    on_context_initializing([[maybe_unused]] const ExtensionContainer& extension_container) {}

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

    virtual void on_instance_created(const InstanceHandle& /*unused*/) {}

    /**
     * Vote against a physical device by returning false. The context attemps to select a physical
     * device that is accepted by most extensions, meaning this is not a hard criteria and you
     * should check support in "extension_supported" again!
     *
     * The default implementation checks if the desired device extensions are supported.
     */
    virtual bool
    accept_physical_device([[maybe_unused]] const vk::PhysicalDevice& physical_device,
                           [[maybe_unused]] const vk::PhysicalDeviceProperties2& props) {
        bool all_extensions_supported = true;
        const auto supported_extensions = physical_device.enumerateDeviceExtensionProperties();
        for (const auto& required_extension : required_device_extension_names(physical_device)) {
            bool found = false;
            for (const auto& supported_extension : supported_extensions) {
                if (strcmp(supported_extension.extensionName, required_extension) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                all_extensions_supported = false;
                break;
            }
        }

        return all_extensions_supported;
    }

    /* Append a structure to pNext of a getProperties2() call. This can be used to determine
     * extension support.
     *
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* pnext_get_properties_2(void* const p_next) {
        return p_next;
    }

    /* Append a structure to pNext of a getFeatures2() call. This can be used to determine extension
     * support.
     *
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* pnext_get_features_2(void* const p_next) {
        return p_next;
    }

    /* Called after the physical device was select and before extensions are checked for
     * compatibility and check_support is called.*/
    virtual void on_physical_device_selected(const PhysicalDeviceHandle& /*unused*/) {}

    /* E.g. to dismiss a queue that does not support present-to-surface. Similar to
     * accpet_physical_device, the context attempt to select a graphics queue that is accepted by
     * most extensions.
     */
    virtual bool accept_graphics_queue([[maybe_unused]] const vk::Instance& instance,
                                       [[maybe_unused]] const PhysicalDevice& physical_device,
                                       [[maybe_unused]] std::size_t queue_family_index) {
        return true;
    }

    /* Custom check for compatibility after the physical device is ready and queue family indices
     * are determined. At this time the structs wired up in pnext_get_features_2 is valid, you can
     * use those to check if features are available.
     *
     * If this method returns false, it is guaranteed that on_unsupported is called.
     */
    virtual bool extension_supported([[maybe_unused]] const vk::Instance& instance,
                                     [[maybe_unused]] const PhysicalDevice& physical_device,
                                     [[maybe_unused]] const ExtensionContainer& extension_container,
                                     [[maybe_unused]] const QueueInfo& queue_info) {
        return true;
    }

    /**
     * Called when extension support is confirmed for all extensions. Can be used to communicate
     * with other extensions.
     */
    virtual void
    on_extension_support_confirmed([[maybe_unused]] const ExtensionContainer& extension_container) {
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

    virtual void on_device_created(const DeviceHandle& /*unused*/) {}

    /* Called right before context constructor returns. */
    virtual void
    on_context_created([[maybe_unused]] const ContextHandle& context,
                       [[maybe_unused]] const ExtensionContainer& extension_container) {}

    // Called by context if extension was determined as unsupported. The extension might not receive
    // further callbacks.
    virtual void on_unsupported([[maybe_unused]] const std::string& reason) {
        spdlog::warn("extension {} not supported ({})", name, reason);
    }

    // OTHER

    // return strings that should be defined when compiling shaders with Merians shader compiler.
    // Note that device, instance and Merian context extensions are automatically defined as
    // MERIAN_DEVICE_EXT_ENABLED_<NAME>, MERIAN_INSTANCE_EXT_ENABLED_<NAME>,
    // MERIAN_CONTEXT_EXT_ENABLED_
    virtual std::map<std::string, std::string> shader_macro_definitions() {
        return {};
    }

  public:
    const std::string name;
};

} // namespace merian
