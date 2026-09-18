#pragma once

#include "merian/io/video/video_decoder.hpp"
#include "merian/io/video/video_encoder.hpp"

#include <filesystem>

namespace merian {

class VideoDevice;
using VideoDeviceHandle = std::shared_ptr<VideoDevice>;

// Hardware video en-/decoding; frames stay device images.
class VideoDevice {
  public:
    virtual ~VideoDevice() = default;

    // Throws runtime_error if the file cannot be opened or has no hardware-decodable video stream.
    virtual VideoDecoderHandle open_decoder(const std::filesystem::path& path) const = 0;

    // Throws runtime_error if the encoder or the container cannot be set up. The container format
    // follows the file extension.
    virtual VideoEncoderHandle open_encoder(const std::filesystem::path& path,
                                            const VideoEncoderInfo& info) const = 0;
};

} // namespace merian
