#pragma once

#include "merian/utils/configuration.hpp"
#include "merian-nodes/graph/node_io.hpp"

#include <memory>

namespace merian {

class Graph;
class GraphRun;

class Node : public std::enable_shared_from_this<Node> {
  public:
    struct NodeStatus {
        // If this is true the Graph is forced to rebuild before the next run.
        bool request_rebuild{false};
        // If this is true the Graph is free to not call cmd_process on the next run.
        bool skip_run{false};
        // By default FrameData is reset when the graph rebuilds.
        // If set to true, the FrameData is persisted
        bool persist_frame_data{false};
    };

    struct FrameData {};

    static inline NodeOutputDescriptorImage FEEDBACK_OUTPUT_IMAGE{{}, {}, {}, {}, {}};
    static inline NodeOutputDescriptorBuffer FEEDBACK_OUTPUT_BUFFER{{}, {}, {}, {}};

  public:
    virtual ~Node() {}

    virtual std::string name() = 0;

    // A factory for the frame data (data has to exist once per frame-in-flight) for this node.
    // It is quaranteed that cmd_run gets a shared_ptr to frame data that was generated using this
    // method, meaning it can be safely casted.
    virtual std::shared_ptr<FrameData> create_frame_data() {
        return {};
    }

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

    // Called everytime before the graph is run.
    // If rebuild is requested here the graph must rebuild itself before calling cmd_process.
    // Note, that this method is necessarily called again after the rebuild.
    virtual void pre_process([[maybe_unused]] const uint64_t& iteration,
                             [[maybe_unused]] NodeStatus& status) {}

    // Called when the graph is build or rebuild. You get your inputs and outputs for each set_index
    // (see cmd_process), use these to create your descriptor sets and such. You can also make and
    // such uploads here. You should only write to output images that were declared as 'persistent',
    // these are also the same in each set.
    // No not access or modify input images.
    virtual void cmd_build([[maybe_unused]] const vk::CommandBuffer& cmd,
                           [[maybe_unused]] const std::vector<NodeIO>& ios) {}

    // This is called once per iteration.
    // You do not need to insert barriers for node inputs and outputs.
    // Use the descriptor set according to set_index (-1 if there are no sets).
    // If you need to perform layout transistions use the barrier() methods of the images.
    // `run.get_iteration()` counts the iterations since last build.
    virtual void cmd_process([[maybe_unused]] const vk::CommandBuffer& cmd,
                             [[maybe_unused]] GraphRun& run,
                             [[maybe_unused]] const std::shared_ptr<FrameData>& frame_data,
                             [[maybe_unused]] const uint32_t set_index,
                             [[maybe_unused]] const NodeIO& io) {}

    // Declare your configuration options.
    virtual void get_configuration([[maybe_unused]] Configuration& config,
                                   [[maybe_unused]] bool& needs_rebuild) {}
};

using NodeHandle = std::shared_ptr<Node>;

} // namespace merian
