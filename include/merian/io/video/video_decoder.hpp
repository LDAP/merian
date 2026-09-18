#pragma once

#include "merian/io/video/video_frame.hpp"

#include <chrono>
#include <string>

namespace merian {

struct VideoStreamInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    // 0 if the container does not declare one.
    double frame_rate = 0;
    std::chrono::nanoseconds duration{};
    std::string codec_name;
};

class VideoDecoder;
using VideoDecoderHandle = std::shared_ptr<VideoDecoder>;

// Decodes the video stream of a container with the hardware decoder.
class VideoDecoder {
  public:
    virtual ~VideoDecoder() = default;

    virtual const VideoStreamInfo& get_stream_info() const = 0;

    // Returns nullptr at the end of the stream.
    virtual VideoFrameHandle read_frame() = 0;

    // Restarts decoding at the keyframe at or before `time`.
    virtual void seek(const std::chrono::nanoseconds& time) = 0;
};

} // namespace merian
