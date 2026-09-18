#pragma once

#include "merian/io/video/video_device.hpp"
#include "merian/vk/extension/extension.hpp"

namespace merian {

// Mixin for ContextExtensions that can create a VideoDevice.
class VideoDeviceProvider {
  public:
    using OperationFlags = uint32_t;

    enum OperationFlagBits {
        DECODE = 0b1,
        ENCODE = 0b10,
    };

    virtual ~VideoDeviceProvider() = default;

    // A subset of DECODE | ENCODE. Must not require a created context.
    virtual OperationFlags
    get_supported_operations(const DeviceSupportQueryInfo& query_info) const = 0;

    // Codecs the encoder accepts, most preferred first.
    virtual const std::vector<std::string>& get_encoder_codecs() const = 0;

    // Throws runtime_error if the operations are not supported.
    virtual VideoDeviceHandle create_video_device(const OperationFlags operations) const = 0;
};

} // namespace merian
