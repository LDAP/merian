#pragma once

#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

// Writes to images files. Note that a queue idle is used after each iteration.
class ImageWriteNode : public Node {

  public:
    ImageWriteNode(const SharedContext context,
                   const ResourceAllocatorHandle allocator,
                   const std::string& base_filename);

    virtual ~ImageWriteNode();

    virtual std::string name() override;

    // Declare the inputs that you require
    virtual std::tuple<std::vector<NodeInputDescriptorImage>,
                       std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    virtual void pre_process([[maybe_unused]] const uint64_t& iteration,
                             [[maybe_unused]] NodeStatus& status) override;

    virtual void
    cmd_process([[maybe_unused]] const vk::CommandBuffer& cmd,
                [[maybe_unused]] GraphRun& run,
                [[maybe_unused]] const uint32_t set_index,
                [[maybe_unused]] const std::vector<ImageHandle>& image_inputs,
                [[maybe_unused]] const std::vector<BufferHandle>& buffer_inputs,
                [[maybe_unused]] const std::vector<ImageHandle>& image_outputs,
                [[maybe_unused]] const std::vector<BufferHandle>& buffer_outputs) override;

    virtual void get_configuration([[maybe_unused]] Configuration& config,
                                   bool& needs_rebuild) override;

    void set_on_record_callback(const std::function<void()> callback);

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;

    std::function<void()> on_record_callback;

    std::string base_filename;
    std::vector<char> buf;

    int64_t iteration = 0;
    uint32_t image_index = 0;

    int format = 0;

    bool record_enable = false;
    int record_iteration = 0;
    int trigger_run = -1;

    bool record_next = false;
    bool rebuild_after_capture = false;
    bool rebuild_on_record = false;

    int it_power = 1;
    int it_offset = 0;
};

} // namespace merian
