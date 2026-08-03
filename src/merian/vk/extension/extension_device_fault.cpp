#include "merian/vk/extension/extension_device_fault.hpp"

namespace merian {

DeviceSupportInfo
ExtensionDeviceFault::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return DeviceSupportInfo::check(query_info, {"deviceFault"});
}

void ExtensionDeviceFault::on_device_created(const DeviceHandle& device,
                                             const ExtensionContainer& /*extension_container*/) {
    this->device = device;
}

void ExtensionDeviceFault::dump() const {
    vk::DeviceFaultCountsEXT counts;
    if (const vk::Result result = device->get_device().getFaultInfoEXT(&counts, nullptr);
        result != vk::Result::eSuccess) {
        // the query is only valid on a lost device, so this is the common case for other errors
        SPDLOG_WARN("no device fault info: {}", vk::to_string(result));
        return;
    }

    std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
    std::vector<vk::DeviceFaultVendorInfoEXT> vendor_infos(counts.vendorInfoCount);
    vk::DeviceFaultInfoEXT info;
    info.pAddressInfos = addresses.data();
    info.pVendorInfos = vendor_infos.data();
    if (const vk::Result result = device->get_device().getFaultInfoEXT(&counts, &info);
        result != vk::Result::eSuccess) {
        SPDLOG_ERROR("could not query device fault info: {}", vk::to_string(result));
        return;
    }

    SPDLOG_ERROR("device fault: {}", info.description.data());
    for (const auto& address : addresses) {
        // the fault is in [reportedAddress rounded down to addressPrecision, + addressPrecision)
        SPDLOG_ERROR("  {} at 0x{:x} (precision 0x{:x})", vk::to_string(address.addressType),
                     address.reportedAddress, address.addressPrecision);
    }
    for (const auto& vendor_info : vendor_infos) {
        SPDLOG_ERROR("  vendor: {} (code 0x{:x}, data 0x{:x})", vendor_info.description.data(),
                     vendor_info.vendorFaultCode, vendor_info.vendorFaultData);
    }
}

} // namespace merian
