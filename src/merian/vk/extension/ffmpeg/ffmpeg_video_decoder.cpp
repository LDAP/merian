#include "ffmpeg_video.hpp"

#include "merian/utils/pointer.hpp"

#include <spdlog/spdlog.h>

namespace merian {

namespace {

AVPixelFormat select_hw_format(AVCodecContext* codec_context, const AVPixelFormat* formats) {
    for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; format++) {
        if (*format != AV_PIX_FMT_VULKAN) {
            continue;
        }

        AVBufferRef* frames_ref = nullptr;
        const int result = avcodec_get_hw_frames_parameters(
            codec_context, codec_context->hw_device_ctx, AV_PIX_FMT_VULKAN, &frames_ref);
        if (result < 0) {
            SPDLOG_ERROR("could not derive the decoder frame pool: {}", av_error(result));
            return AV_PIX_FMT_NONE;
        }

        auto* frames = reinterpret_cast<AVHWFramesContext*>(frames_ref->data);
        auto* vulkan_frames = static_cast<AVVulkanFramesContext*>(frames->hwctx);
        // per-plane views of the multiplanar decode output need a mutable format, and the plane
        // formats do not carry the parent's usage
        vulkan_frames->img_flags |= VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                                    VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        vulkan_frames->usage = static_cast<VkImageUsageFlagBits>(
            vulkan_frames->usage | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        const int init = av_hwframe_ctx_init(frames_ref);
        if (init < 0) {
            SPDLOG_ERROR("could not initialize the decoder frame pool: {}", av_error(init));
            av_buffer_unref(&frames_ref);
            return AV_PIX_FMT_NONE;
        }
        av_buffer_unref(&codec_context->hw_frames_ctx);
        codec_context->hw_frames_ctx = frames_ref;
        return AV_PIX_FMT_VULKAN;
    }

    SPDLOG_ERROR("the decoder does not offer Vulkan frames");
    return AV_PIX_FMT_NONE;
}

} // namespace

FFmpegVideoDecoder::FFmpegVideoDecoder(const VideoDeviceHandle& device,
                                       const std::filesystem::path& path)
    : device(device), context(debugable_ptr_cast<FFmpegVideoDevice>(device)->get_context()) {
    const auto fail = [&](const std::string& message) {
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        throw std::runtime_error{message};
    };

    int result = avformat_open_input(&format_context, path.string().c_str(), nullptr, nullptr);
    if (result < 0) {
        throw std::runtime_error{
            fmt::format("could not open {}: {}", path.string(), av_error(result))};
    }
    result = avformat_find_stream_info(format_context, nullptr);
    if (result < 0) {
        fail(fmt::format("no stream info in {}: {}", path.string(), av_error(result)));
    }

    const AVCodec* codec = nullptr;
    stream_index = av_find_best_stream(format_context, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (stream_index < 0) {
        fail(fmt::format("no video stream in {}", path.string()));
    }

    bool vulkan_supported = false;
    for (int i = 0;; i++) {
        const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
        if (config == nullptr) {
            break;
        }
        vulkan_supported |= (config->device_type == AV_HWDEVICE_TYPE_VULKAN) &&
                            ((config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) != 0);
    }
    if (!vulkan_supported) {
        fail(fmt::format("{} has no Vulkan hardware decoder", avcodec_get_name(codec->id)));
    }

    const AVStream* stream = format_context->streams[stream_index];
    codec_context = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_context, stream->codecpar);
    codec_context->hw_device_ctx =
        av_buffer_ref(debugable_ptr_cast<FFmpegVideoDevice>(device)->get_av_hw_device_context());
    codec_context->get_format = [](AVCodecContext* av_context, const AVPixelFormat* formats) {
        return select_hw_format(av_context, formats);
    };
    codec_context->pkt_timebase = stream->time_base;

    result = avcodec_open2(codec_context, codec, nullptr);
    if (result < 0) {
        fail(fmt::format("could not open the decoder: {}", av_error(result)));
    }

    stream_info.width = static_cast<uint32_t>(codec_context->width);
    stream_info.height = static_cast<uint32_t>(codec_context->height);
    stream_info.frame_rate = stream->avg_frame_rate.den != 0 ? av_q2d(stream->avg_frame_rate) : 0;
    stream_info.codec_name = avcodec_get_name(codec->id);
    if (stream->duration != AV_NOPTS_VALUE) {
        stream_info.duration = std::chrono::nanoseconds{
            av_rescale_q(stream->duration, stream->time_base, AVRational{1, 1000000000})};
    }

    SPDLOG_INFO("decoding {} ({} {}x{} @ {:.3f} fps)", path.string(), stream_info.codec_name,
                stream_info.width, stream_info.height, stream_info.frame_rate);
}

FFmpegVideoDecoder::~FFmpegVideoDecoder() {
    avcodec_free_context(&codec_context);
    avformat_close_input(&format_context);
}

VideoFrameHandle FFmpegVideoDecoder::read_frame() {
    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();

    while (true) {
        const int result = avcodec_receive_frame(codec_context, frame);
        if (result == 0) {
            av_packet_free(&packet);
            if (frame->time_base.num == 0) {
                frame->time_base = format_context->streams[stream_index]->time_base;
            }
            if (frame->pts == AV_NOPTS_VALUE) {
                frame->pts = frame->best_effort_timestamp;
            }
            return FFmpegVideoFrame::create(context, frame, vk::ImageUsageFlagBits::eSampled);
        }
        if (result != AVERROR(EAGAIN)) {
            // AVERROR_EOF or a decode failure: nothing more will come out
            if (result != AVERROR_EOF) {
                SPDLOG_ERROR("decode failed: {}", av_error(result));
            }
            break;
        }
        if (end_of_stream) {
            break;
        }

        int read = 0;
        do {
            av_packet_unref(packet);
            read = av_read_frame(format_context, packet);
        } while (read >= 0 && packet->stream_index != stream_index);

        if (read < 0) {
            end_of_stream = true;
            avcodec_send_packet(codec_context, nullptr);
        } else {
            const int sent = avcodec_send_packet(codec_context, packet);
            if (sent < 0) {
                SPDLOG_ERROR("could not submit a packet to the decoder: {}", av_error(sent));
                break;
            }
        }
    }

    av_packet_free(&packet);
    av_frame_free(&frame);
    return nullptr;
}

void FFmpegVideoDecoder::seek(const std::chrono::nanoseconds& time) {
    const AVStream* stream = format_context->streams[stream_index];
    const int64_t timestamp =
        av_rescale_q(time.count(), AVRational{1, 1000000000}, stream->time_base);
    if (av_seek_frame(format_context, stream_index, timestamp, AVSEEK_FLAG_BACKWARD) < 0) {
        SPDLOG_WARN("seek to {:.3f}s failed", std::chrono::duration<double>(time).count());
        return;
    }
    avcodec_flush_buffers(codec_context);
    end_of_stream = false;
}

} // namespace merian
