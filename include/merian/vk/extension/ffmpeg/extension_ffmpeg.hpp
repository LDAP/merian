#pragma once

#include "merian/io/video/video_device_provider.hpp"
#include "merian/vk/extension/extension.hpp"

namespace merian {

// Hardware video en-/decoding through FFmpeg, running on merian's own Vulkan device.
class ExtensionFFmpeg : public ContextExtension, public VideoDeviceProvider {
  public:
    static constexpr const char* name = "FFmpeg";

    ExtensionFFmpeg();

    ~ExtensionFFmpeg() override;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    std::vector<QueueRequest> request_queues(const PhysicalDeviceHandle& physical_device) override;

    void on_context_created(const ContextHandle& context,
                            const ExtensionContainer& extension_container) override;

    // ----------------------------------------

    OperationFlags
    get_supported_operations(const DeviceSupportQueryInfo& query_info) const override;

    const std::vector<std::string>& get_encoder_codecs() const override;

    VideoDeviceHandle create_video_device(const OperationFlags operations) const override;

  private:
    std::weak_ptr<Context> context;
    std::vector<std::string> encoder_codecs;
    // Shared, so FFmpeg does not hand the same video queue to two callers.
    mutable std::weak_ptr<VideoDevice> video_device;
};

} // namespace merian
