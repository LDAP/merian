#pragma once

#include "merian/io/video/video_frame.hpp"

#include <chrono>
#include <string>

namespace merian {

struct VideoEncoderInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    double frame_rate = 30;
    // One of VideoDeviceProvider::get_encoder_codecs().
    std::string codec;
    // 0 lets the encoder choose.
    int64_t bit_rate = 0;
    // Encode 10 bit instead of 8 bit.
    bool high_bit_depth = false;
};

class VideoEncoder;
using VideoEncoderHandle = std::shared_ptr<VideoEncoder>;

// Encodes frames with the hardware encoder and muxes them into a container. Destroying the
// encoder finishes the file.
class VideoEncoder {
  public:
    virtual ~VideoEncoder() = default;

    // nullptr if the pool is exhausted or the file was already finished.
    virtual VideoFrameHandle acquire_frame() = 0;

    // The caller must have signalled the frame's semaphore. Frames are encoded in the order they
    // were acquired, whatever order they are submitted in. Safe to call from another thread.
    virtual void submit_frame(const VideoFrameHandle& frame) = 0;

    // Further submits are ignored.
    virtual void finish() = 0;

    virtual uint64_t get_frame_count() const = 0;
};

} // namespace merian
