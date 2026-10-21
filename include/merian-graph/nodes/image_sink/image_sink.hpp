#pragma once

#include "merian-graph/connectors/image/vk_image_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/graph/time_provider.hpp"
#include "merian-graph/nodes/utils.hpp"
#include "merian/utils/chrono.hpp"
#include "merian/utils/enums.hpp"
#include "merian/utils/stopwatch.hpp"

namespace merian {

MERIAN_ENUM(ImageSinkTrigger, ITERATION, FRAMETIME, END_OF_RUN)

MERIAN_ENUM(ImageSinkTimeReference, SYSTEM, GRAPH)

class ImageSink : public Node, public TimeProvider {
  public:
    virtual std::vector<InputConnectorDescriptor> describe_inputs() override;

    virtual NodeStatusFlags pre_process(const NodeIO& io, const NodeProcessInfo& info) override;

    [[nodiscard]] virtual NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    virtual NodeStatusFlags properties(Properties& config) override;

    bool provides_time() const override;

    [[nodiscard]] std::optional<std::chrono::nanoseconds> provide_time() override;

    void record(const std::chrono::nanoseconds& current_graph_time);

  private:
    template <typename T>
    void get_format_args(const T& consumer,
                         const vk::Extent3D& input_extent,
                         const vk::Extent3D& output_extent,
                         const uint64_t run_iteration,
                         const uint64_t total_iteration,
                         const std::chrono::nanoseconds& graph_time_since_record,
                         const std::chrono::nanoseconds& graph_time,
                         const std::chrono::nanoseconds& system_time_since_record) {
        consumer(fmt::arg("record_iteration", iteration));
        consumer(fmt::arg("image_index_total", num_captures_since_init));
        consumer(fmt::arg("image_index_record", num_captures_since_record));
        consumer(fmt::arg("run_iteration", run_iteration));
        consumer(fmt::arg("total_iteration", total_iteration));
        consumer(fmt::arg("graph_time", to_seconds(graph_time)));
        consumer(fmt::arg("graph_time_since_record", to_seconds(graph_time_since_record)));
        consumer(fmt::arg("system_time_since_record", to_seconds(system_time_since_record)));
        consumer(fmt::arg("input_width", input_extent.width));
        consumer(fmt::arg("input_height", input_extent.height));
        consumer(fmt::arg("output_width", output_extent.width));
        consumer(fmt::arg("output_height", output_extent.height));
        consumer(fmt::arg("random", rand()));
    }

  private:
    VkImageInHandle con_src = VkImageIn::create();

    std::string filename_format = "image_{image_index_total:04}";

    float scale = 1;
    int64_t iteration = 0;
    uint32_t num_captures_since_init = 0;
    std::chrono::nanoseconds record_graph_time_point;
    Stopwatch record_time_point;

    double last_record_time_millis;
    double last_frame_time_millis;
    bool undersampling = false;

    bool start_stop_record = false;
    int format = 0;
    bool keep_alpha = false;
    bool embed_metadata = true;

    bool record_enable = false;
    int start_at_run = -1;
    ImageSinkTrigger trigger = ImageSinkTrigger::ITERATION;
    int record_iteration = 1;
    int record_iteration_at_start = 1;
    int num_captures_since_record = 0;
    bool reset_record_iteration_at_stop = true;

    ImageSinkTimeReference time_reference = ImageSinkTimeReference::SYSTEM;
    FrameRate record_rate;
    TimeProviderMode time_provider = TimeProviderMode::OFF;
    uint64_t provided_frames = 0;
    // the rate must not change within a timeline
    FrameRate provided_rate;

    bool record_next = false;
    bool rebuild_after_capture = false;
    bool rebuild_on_record = false;

    int it_power = 1;
    int it_offset = 1;

    int stop_at_run = -1;
    int stop_after_iteration = -1;
    float stop_after_seconds = -1;
    int stop_after_num_captures_since_record = -1;
    int exit_at_run = -1;
    int exit_at_iteration = -1;
    float exit_after_seconds = -1;

    bool needs_rebuild = false;
};

} // namespace merian
