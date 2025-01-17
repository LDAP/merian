#pragma once

#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian_nodes {

// Writes to images files.
class ImageWrite : public Node {
    class FrameData {
      public:
        std::optional<ImageHandle> intermediate_image;
    };

  public:
    /**
     * @brief      Constructs a new instance.
     *
     * @param      allocator      The allocator used to create copies of the input.
     */
    ImageWrite(const ContextHandle& context,
               const ResourceAllocatorHandle& allocator,
               const std::string& filename_format = "image_{image_index_total:04}");

    virtual ~ImageWrite();

    virtual std::vector<InputConnectorHandle> describe_inputs() override;

    virtual NodeStatusFlags pre_process(GraphRun& run, const NodeIO& io) override;

    virtual void process(GraphRun& run,
                         const CommandBufferHandle& cmd,
                         const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) override;

    virtual NodeStatusFlags properties(Properties& config) override;

    // Set a callback that can be called on capture or record.
    void set_callback(const std::function<void()>& callback);

    void record(const std::chrono::nanoseconds& current_graph_time);

  private:
    template <typename T>
    void get_format_args(const T& consumer,
                         const vk::Extent3D& input_extent,
                         const vk::Extent3D& output_extent,
                         const uint64_t run_iteration,
                         const std::chrono::nanoseconds& graph_time_since_record,
                         const std::chrono::nanoseconds& graph_time,
                         const std::chrono::nanoseconds& system_time_since_record) {
        consumer(fmt::arg("record_iteration", iteration));
        consumer(fmt::arg("image_index_total", num_captures_since_init));
        consumer(fmt::arg("image_index_record", num_captures_since_record));
        consumer(fmt::arg("run_iteration", run_iteration));
        consumer(fmt::arg("graph_time", to_seconds(graph_time)));
        consumer(fmt::arg("graph_time_since_record", to_seconds(graph_time_since_record)));
        consumer(fmt::arg("system_time_since_record", to_seconds(system_time_since_record)));
        consumer(fmt::arg("input_width", input_extent.width));
        consumer(fmt::arg("input_height", input_extent.height));
        consumer(fmt::arg("output_width", output_extent.width));
        consumer(fmt::arg("output_height", output_extent.height));
        consumer(fmt::arg("random", rand()));
        // backward compat
        consumer(fmt::arg("time", to_seconds(graph_time_since_record)));
        consumer(fmt::arg("width", output_extent.width));
        consumer(fmt::arg("height", output_extent.height));
    }

  private:
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;

    ManagedVkImageInHandle con_src = ManagedVkImageIn::transfer_src("src");

    uint32_t max_concurrent_tasks = std::thread::hardware_concurrency();
    uint32_t concurrent_tasks = 0;
    std::mutex mutex_concurrent;
    std::condition_variable cv_concurrent;

    std::function<void()> callback;

    std::string filename_format;

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

    bool record_enable = false;
    int start_at_run = -1;
    int trigger = 0;
    int record_iteration = 1;
    int record_iteration_at_start = 1;
    int num_captures_since_record = 0;
    bool reset_record_iteration_at_stop = true;

    int time_reference = 0; // system, graph
    float record_framerate = 30;
    float record_frametime_millis = 1000.f / 30.f;

    bool record_next = false;
    bool rebuild_after_capture = false;
    bool rebuild_on_record = false;
    bool callback_after_capture = false;
    bool callback_on_record = false;

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

} // namespace merian_nodes
