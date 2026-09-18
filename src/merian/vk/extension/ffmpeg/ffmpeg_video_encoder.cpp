#include "ffmpeg_video.hpp"

#include "merian/utils/defer.hpp"
#include "merian/utils/pointer.hpp"

#include <spdlog/spdlog.h>

namespace merian {

FFmpegVideoEncoder::FFmpegVideoEncoder(const VideoDeviceHandle& device,
                                       const std::filesystem::path& path,
                                       const VideoEncoderInfo& info)
    : device(device), context(debugable_ptr_cast<FFmpegVideoDevice>(device)->get_context()),
      path(path) {
    const auto fail = [&](const std::string& message) {
        av_buffer_unref(&frames_context);
        avcodec_free_context(&codec_context);
        if (format_context != nullptr) {
            avio_closep(&format_context->pb);
            avformat_free_context(format_context);
            format_context = nullptr;
        }
        throw std::runtime_error{message};
    };

    const AVCodec* codec = avcodec_find_encoder_by_name(info.codec.c_str());
    if (codec == nullptr) {
        fail(fmt::format("unknown encoder {}", info.codec));
    }

    int result =
        avformat_alloc_output_context2(&format_context, nullptr, nullptr, path.string().c_str());
    if (result < 0) {
        fail(fmt::format("no container for {}: {}", path.string(), av_error(result)));
    }

    // frame pool

    frames_context = av_hwframe_ctx_alloc(
        debugable_ptr_cast<FFmpegVideoDevice>(device)->get_av_hw_device_context());
    if (frames_context == nullptr) {
        fail("could not allocate the encoder frame pool");
    }
    auto* frames = reinterpret_cast<AVHWFramesContext*>(frames_context->data);
    frames->format = AV_PIX_FMT_VULKAN;
    frames->sw_format = info.high_bit_depth ? AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;
    frames->width = static_cast<int>(info.width);
    frames->height = static_cast<int>(info.height);
    // enough for the encoder's reference frames plus the frames the caller keeps in flight
    frames->initial_pool_size = 32;

    auto* vulkan_frames = static_cast<AVVulkanFramesContext*>(frames->hwctx);
    // the caller writes the planes through per-plane storage views
    vulkan_frames->img_flags |= VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                                VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    vulkan_frames->usage = static_cast<VkImageUsageFlagBits>(
        vulkan_frames->usage | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    result = av_hwframe_ctx_init(frames_context);
    if (result < 0) {
        fail(fmt::format("could not initialize the encoder frame pool: {}", av_error(result)));
    }

    // encoder

    codec_context = avcodec_alloc_context3(codec);
    if (codec_context == nullptr) {
        fail("could not allocate the encoder");
    }
    codec_context->width = static_cast<int>(info.width);
    codec_context->height = static_cast<int>(info.height);
    codec_context->pix_fmt = AV_PIX_FMT_VULKAN;
    codec_context->sw_pix_fmt = frames->sw_format;
    codec_context->hw_frames_ctx = av_buffer_ref(frames_context);
    codec_context->time_base = AVRational{1, 90000};
    codec_context->framerate = av_d2q(info.frame_rate, 1 << 16);
    frame_duration = av_inv_q(codec_context->framerate);
    codec_context->bit_rate = info.bit_rate;
    codec_context->colorspace = AVCOL_SPC_BT709;
    codec_context->color_primaries = AVCOL_PRI_BT709;
    codec_context->color_trc = AVCOL_TRC_BT709;
    codec_context->color_range = AVCOL_RANGE_MPEG;
    if ((format_context->oformat->flags & AVFMT_GLOBALHEADER) != 0) {
        codec_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    result = avcodec_open2(codec_context, codec, nullptr);
    if (result < 0) {
        fail(fmt::format("could not open {}: {}", info.codec, av_error(result)));
    }

    // container

    stream = avformat_new_stream(format_context, nullptr);
    if (stream == nullptr) {
        fail("could not add a stream to the container");
    }
    avcodec_parameters_from_context(stream->codecpar, codec_context);
    stream->time_base = codec_context->time_base;
    stream->avg_frame_rate = codec_context->framerate;

    if ((format_context->oformat->flags & AVFMT_NOFILE) == 0) {
        std::filesystem::create_directories(std::filesystem::absolute(path).parent_path());
        result = avio_open(&format_context->pb, path.string().c_str(), AVIO_FLAG_WRITE);
        if (result < 0) {
            fail(fmt::format("could not open {}: {}", path.string(), av_error(result)));
        }
    }
    result = avformat_write_header(format_context, nullptr);
    if (result < 0) {
        fail(fmt::format("could not write the container header: {}", av_error(result)));
    }

    SPDLOG_INFO("encoding {} ({} {}x{} @ {:.3f} fps)", path.string(), info.codec, info.width,
                info.height, info.frame_rate);
}

FFmpegVideoEncoder::~FFmpegVideoEncoder() {
    finish();
    av_buffer_unref(&frames_context);
    avcodec_free_context(&codec_context);
    if (format_context != nullptr) {
        avio_closep(&format_context->pb);
        avformat_free_context(format_context);
    }
}

VideoFrameHandle FFmpegVideoEncoder::acquire_frame() {
    const std::lock_guard lock{mutex};
    if (finished) {
        return nullptr;
    }

    AVFrame* frame = av_frame_alloc();
    const int result = av_hwframe_get_buffer(frames_context, frame, 0);
    if (result < 0) {
        SPDLOG_ERROR("could not acquire an encoder frame: {}", av_error(result));
        av_frame_free(&frame);
        return nullptr;
    }
    frame->time_base = codec_context->time_base;
    frame->colorspace = codec_context->colorspace;
    frame->color_primaries = codec_context->color_primaries;
    frame->color_trc = codec_context->color_trc;
    frame->color_range = codec_context->color_range;

    const VideoFrameHandle acquired_frame =
        FFmpegVideoFrame::create(context, frame, vk::ImageUsageFlagBits::eStorage);
    if (acquired_frame) {
        debugable_ptr_cast<FFmpegVideoFrame>(acquired_frame)->acquire_order = acquired++;
    }
    return acquired_frame;
}

void FFmpegVideoEncoder::submit_frame(const VideoFrameHandle& frame) {
    const std::lock_guard lock{mutex};
    if (finished) {
        return;
    }
    defer {
        encoded.notify_all();
    };

    out_of_order.emplace(debugable_ptr_cast<FFmpegVideoFrame>(frame)->acquire_order, frame);
    for (auto it = out_of_order.find(next_to_encode); it != out_of_order.end();
         it = out_of_order.find(next_to_encode)) {
        const auto ready = debugable_ptr_cast<FFmpegVideoFrame>(it->second);
        ready->set_presentation_time(std::chrono::nanoseconds{av_rescale_q(
            static_cast<int64_t>(next_to_encode), frame_duration, AVRational{1, 1000000000})});
        next_to_encode++;
        out_of_order.erase(it);

        const int result = avcodec_send_frame(codec_context, ready->get_av_frame());
        if (result < 0) {
            SPDLOG_ERROR("could not submit a frame to the encoder: {}", av_error(result));
            return;
        }
        frame_count++;
        drain();
    }
}

void FFmpegVideoEncoder::drain() {
    AVPacket* packet = av_packet_alloc();
    while (true) {
        const int result = avcodec_receive_packet(codec_context, packet);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
            break;
        }
        if (result < 0) {
            SPDLOG_ERROR("could not read a packet from the encoder: {}", av_error(result));
            break;
        }
        packet->stream_index = stream->index;
        av_packet_rescale_ts(packet, codec_context->time_base, stream->time_base);
        const int written = av_interleaved_write_frame(format_context, packet);
        if (written < 0) {
            SPDLOG_ERROR("could not mux a packet: {}", av_error(written));
            break;
        }
    }
    av_packet_free(&packet);
}

void FFmpegVideoEncoder::finish() {
    std::unique_lock lock{mutex};
    if (finished) {
        return;
    }
    // frames handed out but not submitted yet are still in flight on the GPU
    if (!encoded.wait_for(lock, std::chrono::seconds{5},
                          [this] { return next_to_encode == acquired; })) {
        SPDLOG_WARN("finishing with {} frames still in flight", acquired - next_to_encode);
    }
    finished = true;

    // FFmpeg's Vulkan encoder crashes when flushed without ever having received a frame
    if (frame_count > 0) {
        avcodec_send_frame(codec_context, nullptr);
        drain();
    }
    av_write_trailer(format_context);
    SPDLOG_INFO("wrote {} frames to {}", frame_count.load(), path.string());
}

} // namespace merian
