#include "ffmpeg_video.hpp"

#include <spdlog/spdlog.h>

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace merian {

namespace {

// Kr, Kb of the luma coefficients; ITU-R BT.601 / BT.709 / BT.2020. Untagged streams follow the
// convention that standard definition is BT.601 and everything above is BT.709.
std::pair<float, float> luma_coefficients(const AVColorSpace color_space, const int height) {
    switch (color_space) {
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
        return {0.299f, 0.114f};
    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:
        return {0.2627f, 0.0593f};
    case AVCOL_SPC_UNSPECIFIED:
        return height <= 576 ? std::pair{0.299f, 0.114f} : std::pair{0.2126f, 0.0722f};
    default:
        return {0.2126f, 0.0722f};
    }
}

class CodecImage : public Image {
  public:
    CodecImage(const ContextHandle& context,
               const vk::Image& image,
               const vk::ImageCreateInfo& create_info,
               const vk::ImageLayout current_layout)
        : Image(context, image, create_info, current_layout) {}

    ~CodecImage() override {
        get_image() = VK_NULL_HANDLE;
    }
};

} // namespace

VideoFrameHandle FFmpegVideoFrame::create(const ContextHandle& context,
                                          AVFrame* frame,
                                          const vk::ImageUsageFlags plane_usage) {
    const auto* frames = reinterpret_cast<AVHWFramesContext*>(frame->hw_frames_ctx->data);
    const auto* vulkan_frame = reinterpret_cast<const AVVkFrame*>(frame->data[0]);

    if (av_pix_fmt_count_planes(frames->sw_format) != 2) {
        SPDLOG_ERROR("unsupported hardware frame format {}",
                     av_get_pix_fmt_name(frames->sw_format));
        av_frame_free(&frame);
        return nullptr;
    }
    if (vulkan_frame->img[1] != VK_NULL_HANDLE) {
        SPDLOG_ERROR("the codec split the frame across multiple images");
        av_frame_free(&frame);
        return nullptr;
    }
    return std::make_shared<FFmpegVideoFrame>(context, frame, plane_usage);
}

FFmpegVideoFrame::FFmpegVideoFrame(const ContextHandle& context,
                                   AVFrame* frame,
                                   const vk::ImageUsageFlags plane_usage)
    : frame(frame) {
    const auto* frames = reinterpret_cast<AVHWFramesContext*>(frame->hw_frames_ctx->data);
    const auto* vulkan_frame = reinterpret_cast<const AVVkFrame*>(frame->data[0]);
    const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(frames->sw_format);

    extent =
        vk::Extent3D{static_cast<uint32_t>(frame->width), static_cast<uint32_t>(frame->height), 1};
    presentation_time = frame->pts == AV_NOPTS_VALUE
                            ? std::chrono::nanoseconds{}
                            : std::chrono::nanoseconds{av_rescale_q(frame->pts, frame->time_base,
                                                                    AVRational{1, 1000000000})};

    // color conversion

    const auto [k_r, k_b] = luma_coefficients(frame->colorspace, frame->height);
    const float k_g = 1.f - k_r - k_b;

    conversion.matrix = glm::mat3{
        glm::vec3{1.f, 1.f, 1.f},
        glm::vec3{0.f, -k_b * 2.f * (1.f - k_b) / k_g, 2.f * (1.f - k_b)},
        glm::vec3{2.f * (1.f - k_r), -k_r * 2.f * (1.f - k_r) / k_g, 0.f},
    };

    // the codecs pack N-bit samples into the most significant bits of a 16 bit plane
    const uint32_t bit_depth = descriptor->comp[0].depth;
    const float depth_scale = bit_depth <= 8 ? 1.f
                                             : 65535.f / static_cast<float>(((1u << bit_depth) - 1u)
                                                                            << (16u - bit_depth));

    glm::vec3 gain{1.f};
    glm::vec3 offset{0.f, 0.5f, 0.5f};
    if (frame->color_range != AVCOL_RANGE_JPEG) {
        // studio swing: luma 16..235, chroma 16..240 of the full range
        gain = glm::vec3{255.f / 219.f, 255.f / 224.f, 255.f / 224.f};
        offset = glm::vec3{16.f / 255.f, 128.f / 255.f, 128.f / 255.f};
    }
    conversion.scale = depth_scale * gain;
    conversion.bias = -offset * gain;
    inverse_conversion.matrix = glm::inverse(conversion.matrix);
    inverse_conversion.scale = 1.f / conversion.scale;
    inverse_conversion.bias = -conversion.bias / conversion.scale;

    // plane views

    const VkFormat* plane_formats = av_vkfmt_from_pixfmt(frames->sw_format);
    const vk::ImageCreateInfo create_info{
        {},
        vk::ImageType::e2D,
        static_cast<vk::Format>(plane_formats[0]),
        extent,
        1,
        1,
        vk::SampleCountFlagBits::e1,
        static_cast<vk::ImageTiling>(vulkan_frame->tiling),
        plane_usage,
    };
    image = std::make_shared<CodecImage>(context, vulkan_frame->img[0], create_info,
                                         static_cast<vk::ImageLayout>(vulkan_frame->layout[0]));

    // the plane formats do not support every usage the parent image carries
    const vk::ImageViewUsageCreateInfo view_usage{plane_usage};
    const vk::ImageAspectFlagBits aspects[] = {vk::ImageAspectFlagBits::ePlane0,
                                               vk::ImageAspectFlagBits::ePlane1};
    for (uint32_t plane = 0; plane < 2; plane++) {
        planes.emplace_back(ImageView::create(
            vk::ImageViewCreateInfo{
                {},
                vulkan_frame->img[0],
                vk::ImageViewType::e2D,
                static_cast<vk::Format>(plane_formats[plane]),
                {},
                vk::ImageSubresourceRange{aspects[plane], 0, 1, 0, 1},
                &view_usage,
            },
            image));
    }
}

FFmpegVideoFrame::~FFmpegVideoFrame() {
    av_frame_free(&frame);
}

const ImageViewHandle& FFmpegVideoFrame::get_plane(const uint32_t index) const {
    return planes[index];
}

const vk::Extent3D& FFmpegVideoFrame::get_extent() const {
    return extent;
}

std::chrono::nanoseconds FFmpegVideoFrame::get_presentation_time() const {
    return presentation_time;
}

const ColorConversion& FFmpegVideoFrame::get_color_conversion() const {
    return conversion;
}

const ColorConversion& FFmpegVideoFrame::get_inverse_color_conversion() const {
    return inverse_conversion;
}

void FFmpegVideoFrame::set_presentation_time(const std::chrono::nanoseconds& time) {
    presentation_time = time;
    frame->pts = av_rescale_q(time.count(), AVRational{1, 1000000000}, frame->time_base);
}

vk::Semaphore FFmpegVideoFrame::get_semaphore() const {
    return reinterpret_cast<const AVVkFrame*>(frame->data[0])->sem[0];
}

uint64_t FFmpegVideoFrame::get_wait_value() const {
    return reinterpret_cast<const AVVkFrame*>(frame->data[0])->sem_value[0];
}

uint64_t FFmpegVideoFrame::take_signal_value(const vk::ImageLayout layout) {
    auto* vulkan_frame = reinterpret_cast<AVVkFrame*>(frame->data[0]);
    vulkan_frame->layout[0] = static_cast<VkImageLayout>(layout);
    // The codec builds its acquire barrier on a video queue, which cannot express shader access,
    // and the semaphore already makes the accesses available.
    vulkan_frame->access[0] = static_cast<VkAccessFlagBits>(VK_ACCESS_NONE);
    return ++vulkan_frame->sem_value[0];
}

vk::ImageLayout FFmpegVideoFrame::get_layout() const {
    return static_cast<vk::ImageLayout>(
        reinterpret_cast<const AVVkFrame*>(frame->data[0])->layout[0]);
}

vk::AccessFlags FFmpegVideoFrame::get_access() const {
    return static_cast<vk::AccessFlags>(
        reinterpret_cast<const AVVkFrame*>(frame->data[0])->access[0]);
}

} // namespace merian
