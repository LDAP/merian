#pragma once

#include "merian/io/video/video_device.hpp"
#include "merian/io/video/video_device_provider.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/utils/vulkan_features.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

namespace merian {

std::string av_error(const int code);

// Everything FFmpeg may use if it is enabled, filtered to what the device supports.
std::vector<const char*> ffmpeg_device_extensions(const PhysicalDeviceHandle& physical_device);

// Wraps one hardware frame. The plane views alias the image the codec owns and are destroyed with
// the wrapper.
class FFmpegVideoFrame : public VideoFrame {
  public:
    // Takes ownership of `frame`. Returns nullptr for layouts other than two-plane 4:2:0.
    static VideoFrameHandle
    create(const ContextHandle& context, AVFrame* frame, const vk::ImageUsageFlags plane_usage);

    FFmpegVideoFrame(const ContextHandle& context,
                     AVFrame* frame,
                     const vk::ImageUsageFlags plane_usage);

    ~FFmpegVideoFrame() override;

    const ImageViewHandle& get_plane(const uint32_t index) const override;
    const vk::Extent3D& get_extent() const override;
    std::chrono::nanoseconds get_presentation_time() const override;
    const ColorConversion& get_color_conversion() const override;
    const ColorConversion& get_inverse_color_conversion() const override;
    vk::Semaphore get_semaphore() const override;
    uint64_t get_wait_value() const override;
    uint64_t take_signal_value(const vk::ImageLayout layout) override;
    vk::ImageLayout get_layout() const override;
    vk::AccessFlags get_access() const override;

    AVFrame* get_av_frame() const {
        return frame;
    }

    void set_presentation_time(const std::chrono::nanoseconds& time);

    uint64_t acquire_order = 0;

  private:
    AVFrame* frame;
    ImageHandle image;
    std::vector<ImageViewHandle> planes;
    vk::Extent3D extent;
    std::chrono::nanoseconds presentation_time;
    ColorConversion conversion;
    ColorConversion inverse_conversion;
};

class FFmpegVideoDevice;

class FFmpegVideoDecoder : public VideoDecoder {
  public:
    FFmpegVideoDecoder(const VideoDeviceHandle& device, const std::filesystem::path& path);

    ~FFmpegVideoDecoder() override;

    const VideoStreamInfo& get_stream_info() const override {
        return stream_info;
    }

    VideoFrameHandle read_frame() override;

    void seek(const std::chrono::nanoseconds& time) override;

  private:
    // FFmpeg reaches back into the device from its queue callbacks
    const VideoDeviceHandle device;
    const ContextHandle context;

    AVFormatContext* format_context = nullptr;
    AVCodecContext* codec_context = nullptr;
    int stream_index = -1;

    VideoStreamInfo stream_info;
    bool end_of_stream = false;
};

class FFmpegVideoEncoder : public VideoEncoder {
  public:
    FFmpegVideoEncoder(const VideoDeviceHandle& device,
                       const std::filesystem::path& path,
                       const VideoEncoderInfo& info);

    ~FFmpegVideoEncoder() override;

    VideoFrameHandle acquire_frame() override;

    void submit_frame(const VideoFrameHandle& frame) override;

    void finish() override;

    uint64_t get_frame_count() const override {
        return frame_count;
    }

  private:
    // Called with `mutex` held.
    void drain();

    const VideoDeviceHandle device;
    const ContextHandle context;
    const std::filesystem::path path;

    AVFormatContext* format_context = nullptr;
    AVCodecContext* codec_context = nullptr;
    AVBufferRef* frames_context = nullptr;
    AVStream* stream = nullptr;

    std::mutex mutex;
    // finish() waits for the frames already handed out
    std::condition_variable encoded;
    std::atomic<uint64_t> frame_count = 0;
    // acquire order, so frames encode in order however their submits complete
    uint64_t acquired = 0;
    uint64_t next_to_encode = 0;
    std::unordered_map<uint64_t, VideoFrameHandle> out_of_order;
    AVRational frame_duration{};
    bool finished = false;
};

class FFmpegVideoDevice : public VideoDevice,
                          public std::enable_shared_from_this<FFmpegVideoDevice> {
  public:
    // Throws runtime_error if the context has no queues for the requested operations.
    FFmpegVideoDevice(const ContextHandle& context,
                      const VideoDeviceProvider::OperationFlags operations);

    ~FFmpegVideoDevice() override;

    VideoDecoderHandle open_decoder(const std::filesystem::path& path) const override;

    VideoEncoderHandle open_encoder(const std::filesystem::path& path,
                                    const VideoEncoderInfo& info) const override;

    const ContextHandle& get_context() const {
        return context;
    }

    AVBufferRef* get_av_hw_device_context() const {
        return hw_device_context;
    }

  private:
    // FFmpeg resolves its entry points through this, so the views it forgets can be destroyed.
    static PFN_vkVoidFunction VKAPI_CALL get_instance_proc_addr(VkInstance instance,
                                                                const char* name);
    static PFN_vkVoidFunction VKAPI_CALL get_device_proc_addr(VkDevice device, const char* name);

    static void lock_queue(AVHWDeviceContext* device_context,
                           const uint32_t queue_family,
                           const uint32_t index);
    static void unlock_queue(AVHWDeviceContext* device_context,
                             const uint32_t queue_family,
                             const uint32_t index);

    const ContextHandle context;
    // FFmpeg submits here too, so everything else has to take the same lock.
    const QueueHandle queue_GCT;

    AVBufferRef* hw_device_context = nullptr;
    // both referenced by the AVVulkanDeviceContext for its lifetime, the features also through
    // the pNext chain built into them
    std::vector<const char*> enabled_extensions;
    VulkanFeatures enabled_features;
};

} // namespace merian
