#include "vk/extension/extension.hpp"

class ExtensionRaytraceQuery : public Extension {
  public:
    ExtensionRaytraceQuery() {}
    ~ExtensionRaytraceQuery() {}
    std::string name() const override {
        return "ExtensionRaytraceQuery";
    }
    std::vector<const char*> required_device_extension_names() const override {
        return {
            // ray query instead of ray pipeline
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // intel doesn't have it pre 2015 (hd 520)
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,     VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,    VK_KHR_RAY_QUERY_EXTENSION_NAME,
        };
    }
};
