#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/graph/node_events.hpp"
#include "merian-graph/graph/time_provider.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"
#include "merian/io/video/video_device_provider.hpp"

#include <filesystem>

namespace merian {

// Encodes a graph output into a video file with the hardware encoder. As a time source it reports
// the time of the frame it is about to encode, so a recording is complete regardless of render
// time.
class VideoWrite : public Node, public TimeProvider {

    struct PushConstant {
        // columns of the RGB -> YCbCr matrix
        glm::vec4 matrix_0{1., 0., 0., 0.};
        glm::vec4 matrix_1{0., 1., 0., 0.};
        glm::vec4 matrix_2{0., 0., 1., 0.};
        glm::vec4 scale{1., 1., 1., 0.};
        // w != 0 encodes the sRGB transfer curve
        glm::vec4 bias{0., 0., 0., 1.};
    };

  public:
    VideoWrite();

    ~VideoWrite() override;

    std::vector<std::string> request_context_extensions() override;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    [[nodiscard]] NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                               const NodeIO& io,
                                               const NodeConnectionInfo& info,
                                               Submission& submission) override;

    [[nodiscard]] NodeStatusFlags pre_process(const NodeIO& io,
                                              const NodeProcessInfo& info) override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

    bool provides_time() const override;

    [[nodiscard]] std::optional<std::chrono::nanoseconds> provide_time() override;

    // Starts a new file. Closes a recording that is still running.
    void record();

    // Finishes the file.
    void stop();

  private:
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    std::optional<ComputeKernel> kernel;
    Versioned<SpecializationInfo> spec_info{MERIAN_SPECIALIZATION_INFO_NONE};

    VideoDeviceHandle device;
    VideoEncoderHandle encoder;

    VkSampledImageInHandle con_src = VkSampledImageIn::create();
    vk::Extent3D extent;
    PushConstant pc;

    NodeEvents events{{
        {"start", [this] { start_stop_record = !record_enable; }},
        {"stop", [this] { start_stop_record = record_enable; }},
        {"toggle", [this] { start_stop_record = true; }},
    }};

    std::string filename_format = "video_{index:03}";
    std::filesystem::path recording_path;
    uint32_t num_recordings = 0;

    int codec_index = 0;
    std::vector<std::string> codecs;
    float framerate = 30;
    float frametime_millis = 1000.f / 30.f;
    int bit_rate_kbit = 0;
    bool high_bit_depth = false;
    bool encode_transfer = true;
    TimeProviderMode time_provider = TimeProviderMode::WHEN_ACTIVE;
    uint64_t provided_frames = 0;
    // the rate must not change within a timeline
    float provided_frametime_millis = 1000.f / 30.f;

    bool record_enable = false;
    bool start_stop_record = false;
    // the content did not advance while the graph time stands still
    std::optional<std::chrono::nanoseconds> last_encoded_time;
    // the encoder only counts a frame once its submission completed
    uint64_t submitted_frames = 0;
    int start_at_run = -1;
    int stop_after_num_frames = -1;
};

} // namespace merian
