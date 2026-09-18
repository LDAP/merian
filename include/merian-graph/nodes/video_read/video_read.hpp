#pragma once

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/graph/node_events.hpp"
#include "merian-graph/graph/time_provider.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"
#include "merian/io/video/video_device_provider.hpp"

#include <filesystem>

namespace merian {

// Decodes a video file with the hardware decoder and converts it to linear RGB. As a time source
// it reports the presentation time of the frame it is about to output.
class VideoRead : public Node, public TimeProvider {

    struct PushConstant {
        // columns of the YCbCr -> RGB matrix
        glm::vec4 matrix_0{1., 0., 0., 0.};
        glm::vec4 matrix_1{0., 1., 0., 0.};
        glm::vec4 matrix_2{0., 0., 1., 0.};
        glm::vec4 scale{1., 1., 1., 0.};
        // w != 0 decodes the sRGB transfer curve
        glm::vec4 bias{0., 0., 0., 1.};
    };

  public:
    VideoRead();

    ~VideoRead() override;

    std::vector<std::string> request_context_extensions() override;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    [[nodiscard]] NodeStatusFlags pre_process(const NodeIO& io,
                                              const NodeProcessInfo& info) override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

    [[nodiscard]] std::optional<std::chrono::nanoseconds> provide_time() override;

  private:
    // Returns nullptr at the end of the stream.
    VideoFrameHandle decode_next();

    // Promotes the next decoded frame. Returns false at the end of the stream.
    bool advance_one();

    std::chrono::nanoseconds presentation_time(const VideoFrameHandle& frame) const;

    // Decodes the next frame if there is none held. False once the stream is exhausted.
    bool ensure_pending();

    void seek_to(const std::chrono::nanoseconds& time);

    // Makes the newest frame with a presentation time at or before `time` the current one.
    void advance_to(const std::chrono::nanoseconds& time);

    ContextHandle context;
    ResourceAllocatorHandle allocator;
    SamplerHandle sampler;
    std::optional<ComputeKernel> kernel;
    Versioned<SpecializationInfo> spec_info{MERIAN_SPECIALIZATION_INFO_NONE};

    VideoDeviceHandle device;
    VideoDecoderHandle decoder;
    VideoFrameHandle current;
    VideoFrameHandle pending;
    // added to every presentation time so the graph clock keeps increasing across loops
    std::chrono::nanoseconds loop_offset{};
    // a reconnect repeats pre_process within the run, the frame is picked only once
    std::optional<uint64_t> advanced_iteration;

    ManagedVkImageOutHandle con_out;
    vk::Extent3D extent;
    PushConstant pc;

    NodeEvents events{{
        {"play", [this] { playing = true; }},
        {"pause", [this] { playing = false; }},
        {"toggle", [this] { playing = !playing; }},
        {"step forward", [this] { step_forward = true; }},
        {"step backward", [this] { step_back = true; }},
        {"restart", [this] { seek_requested = std::chrono::nanoseconds{}; }},
    }};

    std::filesystem::path filename;
    std::string config_filename;
    bool playing = true;
    // advance one frame regardless of the clock
    bool step_forward = false;
    bool step_back = false;
    std::optional<std::chrono::nanoseconds> seek_requested;
    bool reached_end = false;
    float seek_seconds = 0;
    bool loop = true;
    // cleared when the stream turns out not to be loopable
    bool can_loop = true;
    bool decode_transfer = true;
    // Undefined: 16 bit float.
    vk::Format overwrite_format = vk::Format::eUndefined;
};

} // namespace merian
