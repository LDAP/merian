#pragma once

#include "merian/vk/memory/resource_allocations.hpp"

#include <chrono>
#include <glm/glm.hpp>

namespace merian {

// A linear transform between YCbCr samples and RGB for a frame's color space and range:
// forward (decode)  rgb    = matrix * (sample * scale + bias)
// inverse (encode)  sample = (matrix * rgb) * scale + bias
struct ColorConversion {
    glm::mat3 matrix{1.f};
    glm::vec3 scale{1.f};
    glm::vec3 bias{0.f};
};

class VideoFrame;
using VideoFrameHandle = std::shared_ptr<VideoFrame>;

// A hardware video frame in device memory. The plane views alias an image the codec owns, so the
// frame has to stay alive until the accesses submitted for it finished. Wait for get_wait_value()
// before accessing it and signal take_signal_value() afterwards.
class VideoFrame {
  public:
    virtual ~VideoFrame() = default;

    // Luma at index 0, interleaved chroma at index 1.
    virtual const ImageViewHandle& get_plane(const uint32_t index) const = 0;

    virtual const vk::Extent3D& get_extent() const = 0;

    virtual std::chrono::nanoseconds get_presentation_time() const = 0;

    virtual const ColorConversion& get_color_conversion() const = 0;

    virtual const ColorConversion& get_inverse_color_conversion() const = 0;

    virtual vk::Semaphore get_semaphore() const = 0;

    virtual uint64_t get_wait_value() const = 0;

    // Also hands the image over in `layout`, which is where the codec picks it up.
    virtual uint64_t take_signal_value(const vk::ImageLayout layout) = 0;

    virtual vk::ImageLayout get_layout() const = 0;

    virtual vk::AccessFlags get_access() const = 0;
};

} // namespace merian
