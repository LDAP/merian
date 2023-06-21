#pragma once

#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/graph/node_io.hpp"
#include "vulkan/vulkan.hpp"

#include <memory>

namespace merian {

class Graph;

class Node : public std::enable_shared_from_this<Node> {
  public:
    struct NodeStatus {
        // If this is true the Graph is forced to rebuild before the next run.
        bool request_rebuild{false};
        // If this is true the Graph is free to not call cmd_process on the next run.
        bool skip_run{false};
    };

    static inline NodeOutputDescriptorImage FEEDBACK_OUTPUT_IMAGE{{}, {}, {}, {}, {}};
    static inline NodeOutputDescriptorBuffer FEEDBACK_OUTPUT_BUFFER{{}, {}, {}, {}};

  public:
    virtual ~Node() {}

    virtual std::string name() = 0;

    // Declare the inputs that you require
    virtual std::tuple<std::vector<NodeInputDescriptorImage>,
                       std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() {
        return {};
    }

    // Declare your outputs. Based on the output descriptors that were connected to your inputs.
    // You can check format and such here and fail if they are not compatible.
    // This may be called with different parameters when the graph is rebuilding.
    // Note: You do NOT get valid descriptors for delayed images and buffers, since those are
    // instanciated later, instead you get FEEDBACK_OUTPUT_IMAGE and FEEDBACK_OUTPUT_BUFFER
    // respectively.
    virtual std::tuple<std::vector<NodeOutputDescriptorImage>,
                       std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(
        [[maybe_unused]] const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
        [[maybe_unused]] const std::vector<NodeOutputDescriptorBuffer>& connected_buffer_outputs) {
        return {};
    }

    // Called everytime before the graph is run. Can be used to request a rebuild for example.
    virtual void pre_process([[maybe_unused]] NodeStatus& status) {}

    // Called when the graph is build or rebuild. You get your inputs and outputs for each set_index
    // (see cmd_process), use these to create your descriptor sets and such. You can also make and
    // such uploads here. You should only write to output images that were declared as 'persistent',
    // these are also the same in each set.
    // No not access or modify input images.
    virtual void
    cmd_build([[maybe_unused]] const vk::CommandBuffer& cmd,
              [[maybe_unused]] const std::vector<std::vector<ImageHandle>>& image_inputs,
              [[maybe_unused]] const std::vector<std::vector<BufferHandle>>& buffer_inputs,
              [[maybe_unused]] const std::vector<std::vector<ImageHandle>>& image_outputs,
              [[maybe_unused]] const std::vector<std::vector<BufferHandle>>& buffer_outputs) {}

    // This is called once per iteration.
    // You do not need to insert barriers for node inputs and outputs.
    // Use the descriptor set according to set_index.
    // If you need to perform layout transistions use the barrier() methods of the images.
    // `iteration` counts the iterations since last build
    virtual void cmd_process([[maybe_unused]] const vk::CommandBuffer& cmd,
                             [[maybe_unused]] const uint64_t iteration,
                             [[maybe_unused]] const uint32_t set_index,
                             [[maybe_unused]] const std::vector<ImageHandle>& image_inputs,
                             [[maybe_unused]] const std::vector<BufferHandle>& buffer_inputs,
                             [[maybe_unused]] const std::vector<ImageHandle>& image_outputs,
                             [[maybe_unused]] const std::vector<BufferHandle>& buffer_outputs) {}
};

using NodeHandle = std::shared_ptr<Node>;

} // namespace merian
