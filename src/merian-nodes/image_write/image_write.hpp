#pragma once

#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

// Writes to images files.
class ImageWriteNode : public Node {
    class FrameData : public Node::FrameData {
      public:
        std::optional<ImageHandle> intermediate_image;
    };

  public:
    ImageWriteNode(const SharedContext context,
                   const ResourceAllocatorHandle allocator,
                   const std::string& base_filename =
                       "image_{record_iteration:06}_{image_index:06}_{run_iteration:06}");

    virtual ~ImageWriteNode();

    virtual std::string name() override;

    virtual std::shared_ptr<Node::FrameData> create_frame_data() override;

    // Declare the inputs that you require
    virtual std::tuple<std::vector<NodeInputDescriptorImage>,
                       std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    virtual void pre_process([[maybe_unused]] const uint64_t& iteration,
                             [[maybe_unused]] NodeStatus& status) override;

    virtual void cmd_process(const vk::CommandBuffer& cmd,
                             GraphRun& run,
                             const std::shared_ptr<Node::FrameData>& frame_data,
                             const uint32_t set_index,
                             const NodeIO& io) override;

    virtual void get_configuration([[maybe_unused]] Configuration& config,
                                   bool& needs_rebuild) override;

    // Set a callback that can be called on capture or record.
    void set_callback(const std::function<void()> callback);

    void record();

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;

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

    int format = 0;

    bool record_enable = false;
    int record_iteration = 0;
    int trigger_run = -1;

    bool record_next = false;
    bool rebuild_after_capture = false;
    bool rebuild_on_record = false;
    bool callback_after_capture = false;
    bool callback_on_record = false;

    int it_power = 1;
    int it_offset = 0;

    int stop_run = -1;
    int stop_iteration = -1;
    int exit_run = -1;
    int exit_iteration = -1;

    bool needs_rebuild = false;
};

} // namespace merian
