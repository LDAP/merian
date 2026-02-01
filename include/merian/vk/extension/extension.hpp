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
class ContextExtension {

    friend Context;

  public:
    ContextExtension(const std::string& name) : name(name) {}

    virtual ~ContextExtension() = 0;

    // REQUIREMENTS

    /* Extensions that should be enabled instance-wide. The context attempts to enable as many as
     * possible. */
    virtual std::vector<const char*> enable_instance_extension_names(
        const std::unordered_set<std::string>& /*supported_extensions*/) const {
        return {};
    }
    /* Layers that should be enabled instance-wide. The context attempts to enable as many as
     * possible. */
    virtual std::vector<const char*>
    enable_instance_layer_names(const std::unordered_set<std::string>& /*supported_layers*/) const {
        return {};
    }

    /* Extensions that should be enabled device-wide. The context attempts to enable as many as
     * possible. */
    virtual std::vector<const char*>
    enable_device_extension_names(const PhysicalDeviceHandle& /*unused*/) const {
        return {};
    }

    /* Features that should be enabled device-wide (return a featureStructName/featureName pattern).
     * The context attempts to enable as
     * many as possible. */
    virtual std::vector<std::string>
    enable_device_features(const PhysicalDeviceHandle& /*unused*/) const {
        return {};
    }

    // LIFECYCLE (in order)

    /**
     * Notifies the extensions that the context is starting and allows extensions to communicate.
     *
     * Note, this are the desired extensions and support is yet to be determined.
     */
    virtual void
    on_context_initializing([[maybe_unused]] const ExtensionContainer& extension_container,
                            [[maybe_unused]] const vk::detail::DispatchLoaderDynamic& loader) {}

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

    /* Called after the physical device was select and before extensions are checked for
     * compatibility and check_support is called.*/
    virtual void on_physical_device_selected(const PhysicalDeviceHandle& /*unused*/) {}

    /* E.g. to dismiss a queue that does not support present-to-surface. Similar to
     * accpet_physical_device, the context attempt to select a graphics queue that is accepted by
     * most extensions.
     */
    virtual bool accept_graphics_queue([[maybe_unused]] const InstanceHandle& instance,
                                       [[maybe_unused]] const PhysicalDeviceHandle& physical_device,
                                       [[maybe_unused]] std::size_t queue_family_index) {
        return true;
    }

    /* Check if this extension is supported with the supplied instance layers and extensions.
     *
     * If this returned false, then on_unsupported is called.
     *
     * The default implementation returns false if any instance layer or extension is missing.
     */
    virtual bool
    extension_supported(const std::unordered_set<std::string>& supported_instance_extensions,
                        const std::unordered_set<std::string>& supported_instance_layers) {
        for (const auto& req_instance_ext :
             enable_instance_extension_names(supported_instance_extensions)) {
            if (!supported_instance_extensions.contains(req_instance_ext)) {
                SPDLOG_WARN("extension {} requested instance extension {} is not available", name,
                            req_instance_ext);
                return false;
            }
        }
        for (const auto& req_instance_layer :
             enable_instance_layer_names(supported_instance_layers)) {
            if (!supported_instance_layers.contains(req_instance_layer)) {
                SPDLOG_WARN("extension {} requested instance layer {} is not available", name,
                            req_instance_layer);
                return false;
            }
        }

        return true;
    }

    /* Check if this extension is supported with the supplied instance layers, extensions, and
     * device extensions and features.
     *
     * If a device is selected where this function returned false, on_unsupported is called.
     *
     * The default implementation returns false if any instance layer or extension or device
     * extension or feature is missing.
     */
    virtual bool extension_supported(const PhysicalDeviceHandle& physical_device,
                                     [[maybe_unused]] const QueueInfo& queue_info) {

        for (const auto& req_device_ext : enable_device_extension_names(physical_device)) {
            if (!physical_device->extension_supported(req_device_ext)) {
                return false;
            }
        }
        for (const auto& req_device_feature : enable_device_features(physical_device)) {
            if (!physical_device->get_supported_features().get_feature(req_device_feature)) {
                return false;
            }
        }

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

    /**
     * Called with the device create info just before createDevice is called.
     *
     * Use to hook into device creation.
     */
    virtual void on_create_device(const PhysicalDeviceHandle& /*physical_device*/,
                                  VulkanFeatures& /*features*/,
                                  std::vector<const char*>& /*extensions*/) {}

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
