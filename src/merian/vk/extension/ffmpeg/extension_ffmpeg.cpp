#include "merian/vk/extension/ffmpeg/extension_ffmpeg.hpp"

#include "ffmpeg_video.hpp"

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

namespace merian {

namespace {
// Vulkan encoders FFmpeg may offer, most preferred first.
const std::vector<std::string> ENCODER_CODECS = {"hevc_vulkan", "h264_vulkan", "av1_vulkan"};
} // namespace

ExtensionFFmpeg::ExtensionFFmpeg() : ContextExtension() {}

ExtensionFFmpeg::~ExtensionFFmpeg() = default;

DeviceSupportInfo ExtensionFFmpeg::query_device_support(const DeviceSupportQueryInfo& query_info) {
    if (!query_info.queues.supports(vk::QueueFlagBits::eVideoDecodeKHR) &&
        !query_info.queues.supports(vk::QueueFlagBits::eVideoEncodeKHR)) {
        return DeviceSupportInfo{false, "no video queue family"};
    }

    return DeviceSupportInfo::check(
        query_info, {"samplerYcbcrConversion", "synchronization2", "timelineSemaphore"},
        {"shaderImageGatherExtended", "fragmentStoresAndAtomics", "shaderInt64",
         "vertexPipelineStoresAndAtomics", "videoMaintenance1", "videoMaintenance2"},
        {VK_KHR_VIDEO_QUEUE_EXTENSION_NAME}, ffmpeg_device_extensions(query_info.physical_device));
}

std::vector<QueueRequest>
ExtensionFFmpeg::request_queues([[maybe_unused]] const PhysicalDeviceHandle& physical_device) {
    return {
        {.capabilities = vk::QueueFlagBits::eVideoDecodeKHR, .required = false},
        {.capabilities = vk::QueueFlagBits::eVideoEncodeKHR, .required = false},
    };
}

void ExtensionFFmpeg::on_context_created(const ContextHandle& context,
                                         const ExtensionContainer& /*extension_container*/) {
    this->context = context;

    if (context->get_queue_info().supports(vk::QueueFlagBits::eVideoEncodeKHR)) {
        for (const std::string& codec : ENCODER_CODECS) {
            if (avcodec_find_encoder_by_name(codec.c_str()) != nullptr) {
                encoder_codecs.emplace_back(codec);
            }
        }
    }

    SPDLOG_DEBUG("ffmpeg video ready (decode: {}, encoders: [{}])",
                 context->get_queue_info().supports(vk::QueueFlagBits::eVideoDecodeKHR),
                 fmt::join(encoder_codecs, ", "));
}

ExtensionFFmpeg::OperationFlags
ExtensionFFmpeg::get_supported_operations(const DeviceSupportQueryInfo& query_info) const {
    OperationFlags operations = 0;
    if (query_info.queues.supports(vk::QueueFlagBits::eVideoDecodeKHR)) {
        operations |= DECODE;
    }
    if (query_info.queues.supports(vk::QueueFlagBits::eVideoEncodeKHR)) {
        operations |= ENCODE;
    }
    return operations;
}

const std::vector<std::string>& ExtensionFFmpeg::get_encoder_codecs() const {
    return encoder_codecs;
}

VideoDeviceHandle ExtensionFFmpeg::create_video_device(const OperationFlags operations) const {
    const ContextHandle held = context.lock();
    if (!held) {
        throw std::runtime_error{"the context is gone"};
    }
    const OperationFlags supported =
        get_supported_operations(DeviceSupportQueryInfo::from_context(held));
    if ((operations & ~supported) != 0u) {
        throw std::runtime_error{"the device does not support the requested video operations"};
    }

    // one device for everybody, so FFmpeg does not hand the same video queue to two callers
    if (const VideoDeviceHandle shared = video_device.lock()) {
        return shared;
    }
    const auto device = std::make_shared<FFmpegVideoDevice>(held, supported);
    video_device = device;
    return device;
}

} // namespace merian
