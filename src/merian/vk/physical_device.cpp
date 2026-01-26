#include "merian/vk/physical_device.hpp"

namespace merian {
PhysicalDevice::PhysicalDevice(const InstanceHandle& instance,
                               const vk::PhysicalDevice& physical_device)
    : instance(instance), physical_device(physical_device) {

    for (const auto& s_type : get_api_version_property_types(instance->get_vk_api_version())) {
        properties.try_emplace(s_type, get_property(s_type));
    }

    for (const auto& ext : physical_device.enumerateDeviceExtensionProperties()) {
        for (const auto& s_type : get_extension_property_types(ext.extensionName)) {
            properties.try_emplace(s_type, get_property(s_type));
        }
        supported_extensions.emplace(ext.extensionName);
    }

    void* prop_p_next = nullptr;
    for (auto& [_, prop] : properties) {
        prop->set_pnext(prop_p_next);
        prop_p_next = prop->get_structure_ptr();
    }
    physical_device_properties.pNext = prop_p_next;
    physical_device.getProperties2(&physical_device_properties);

    physical_device_memory_properties = physical_device.getMemoryProperties2();

    physical_device_extension_properties = physical_device.enumerateDeviceExtensionProperties();

    void* feat_p_next = nullptr;
    vk::PhysicalDeviceFeatures2* features = nullptr;
    for (const auto& feature : get_all_features()) {
        supported_features.emplace(feature->get_structure_type(), feature);

        if (feature->get_structure_type() == vk::StructureType::ePhysicalDeviceFeatures2) {
            features = reinterpret_cast<vk::PhysicalDeviceFeatures2*>(feature->get_structure_ptr());
            continue;
        }

        bool supported = true;
        for (const auto& req_ext :
             feature->get_required_extensions(instance->get_vk_api_version())) {
            if (!extension_supported(req_ext)) {
                // we assume that if this returns true, all dependent extensions also return true;
                supported = false;
            }
        }

        if (supported) {
            feature->set_pnext(feat_p_next);
            feat_p_next = feature->get_structure_ptr();
        }
    }
    assert(features);
    features->pNext = feat_p_next;
    physical_device.getFeatures2(features);
}

PhysicalDeviceHandle PhysicalDevice::create(const InstanceHandle& instance,
                                            const vk::PhysicalDevice& physical_device) {
    return PhysicalDeviceHandle(new PhysicalDevice(instance, physical_device));
}

} // namespace merian
