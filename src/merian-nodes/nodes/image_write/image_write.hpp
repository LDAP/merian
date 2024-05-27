#pragma once

#include "merian-nodes/connectors/vk_image_in.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian_nodes {

// Writes to images files.
class ImageWriteNode : public Node {
    class FrameData {
      public:
        std::optional<ImageHandle> intermediate_image;
    };

  public:
    /**
     * @brief      Constructs a new instance.
     *
     * @param      allocator      The allocator used to create copies of the input to be able to
     */
    ImageWriteNode(const SharedContext context,
                   const ResourceAllocatorHandle allocator,
                   const std::string& base_filename =
                       "image_{record_iteration:06}_{image_index:06}_{run_iteration:06}");

    virtual ~ImageWriteNode();

    virtual std::vector<InputConnectorHandle> describe_inputs() override;

    virtual NodeStatusFlags pre_process(GraphRun& run, const NodeIO& io) override;

    virtual void process(GraphRun& run,
                         const vk::CommandBuffer& cmd,
                         const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) override;

    virtual NodeStatusFlags configuration(Configuration& config) override;

    // Set a callback that can be called on capture or record.
    void set_callback(const std::function<void()> callback);

    void record();

  private:
    template <typename T>
    void
    get_format_args(const T& consumer, const vk::Extent3D& extent, const uint64_t run_iteration) {
        consumer(fmt::arg("record_iteration", iteration));
        consumer(fmt::arg("image_index", this->image_index++));
        consumer(fmt::arg("run_iteration", run_iteration));
        consumer(fmt::arg("time", time_since_record.millis()));
        consumer(fmt::arg("width", extent.width));
        consumer(fmt::arg("height", extent.height));
        consumer(fmt::arg("random", random()));
    }

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;

    VkImageInHandle con_src = VkImageIn::transfer_src("src");

    uint32_t max_concurrent_tasks = std::thread::hardware_concurrency();
    uint32_t concurrent_tasks = 0;
    std::mutex mutex_concurrent;
    std::condition_variable cv_concurrent;

    std::function<void()> callback;

    std::string filename_format;
    std::vector<char> buf;

    float scale = 1;
    int64_t iteration = 0;
    uint32_t image_index = 0;
    Stopwatch time_since_record;
    double last_record_time_millis;
    double last_frame_time_millis;
    double estimated_frametime_millis = 0;
    bool undersampling = false;

    int format = 0;

    bool record_enable = false;
    int enable_run = -1;
    int trigger = 0;
    int record_iteration = 0;

    float record_framerate = 30;
    float record_frametime_millis = 1000.f / 30.f;

    bool record_next = false;
    bool rebuild_after_capture = false;
    bool rebuild_on_record = false;
    bool callback_after_capture = false;
    bool callback_on_record = false;

    int it_power = 1;
    int it_offset = 0;

    int stop_run = -1;
    int stop_iteration = -1;
    float stop_after_seconds = -1;
    int exit_run = -1;
    int exit_iteration = -1;
    float exit_after_seconds = -1;

    bool needs_rebuild = false;
};

} // namespace merian_nodes
