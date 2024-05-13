#pragma once

#include "connector_input.hpp"
#include "connector_output.hpp"
#include "graph_run.hpp"

#include "merian/utils/configuration.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include <memory>

namespace merian_nodes {

class Node : public std::enable_shared_from_this<Node> {
  public:
    struct InFlightData {};

    using NodeStatusFlags = uint32_t;

    enum NodeStatusFlagBits {
        // Ensures a rebuild before the next call to process(...)
        NEEDS_REBUILD = 0b1,
        // In on_connected: Resets frame data of EVERY frame in flight.
        // In pre_process: Resets frame data of the next run ONLY.
        RESET_FRAME_DATA = 0b10,
    };

  public:
    Node(const std::string& name) : name(name) {}

    virtual ~Node() {}

    // Called each time the graph attempts to connect nodes.
    // If you need to access the resources directly, you need to maintain a copy of the InputHandle.
    //
    // Note that input and output names must be unique.
    virtual std::vector<InputConnectorHandle> describe_inputs() {
        return {};
    }

    // Called each time the graph attempts to connect nodes.
    //
    // If you need to access the resources directly, you need to maintain a copy of the
    // OutputHandle. You won't have access to delayed inputs here, since the corresponding outputs
    // are created later.
    //
    // Note that input and output names must be unique.
    virtual std::vector<OutputConnectorHandle>
    describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
        return {};
    }

    // Called when the graph is fully connected and all inputs and outputs are defined.
    // This is a good place to create layouts and pipelines.
    // This might be called multiple times in the nodes life-cycle (whenever a connection changes).
    // It can be assumed that at the time of calling processing of all in-flight data has finished,
    // that means old pipelines and such can be safely destroyed.
    //
    // The descriptor set layout is automatically constructed from the inputs and outputs.
    // It contains all input and output connectors which get_descriptor_info() method does not
    // return std::nullopt. The order is guaranteed to be all inputs in the order of
    // describe_inputs() then outputs in the order of describe_outputs().
    virtual NodeStatusFlags
    on_connected([[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {
        return {};
    }

    // Called before each run.
    //
    // Note that requesting a rebuild is a heavy operation and should only be called if the outputs
    // change. The graph then has to rebuild itself before calling cmd_process. Note, that this
    // method is called again after the rebuild until no node requests a rebuild.
    //
    // Here you can access the resources for the run or set your own, depending on the descriptor
    // type. It is guaranteed that the descriptor set in process(...) is accordingly updated. If you
    // update resources in process(...) the descriptor set will reflect the changes on iteration
    // later.
    //
    // The supplied command buffer is submitted independently of the flags being returned. Depending
    // on the implementation a separate submit might be used for on_pre_process() and process(),
    // however synchronization between the two is then explicitly ensured.
    virtual NodeStatusFlags pre_process([[maybe_unused]] GraphRun& run,
                                        [[maybe_unused]] const vk::CommandBuffer& cmd) {
        return {};
    }

    // Do your main processing here.
    //
    // You do not need to insert barriers for node inputs and outputs.
    // If you need to perform layout transitions use the barrier() methods of the images.
    // You can provide data that that is required for the current run by setting in_flight_data.
    // The pointer is persisted and supplied again after (graph ring size - 1) runs.
    virtual void process([[maybe_unused]] GraphRun& run,
                         [[maybe_unused]] const vk::CommandBuffer& cmd,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         [[maybe_unused]] std::shared_ptr<InFlightData>& in_flight_data) {}

    // Declare your configuration options and output status information.
    // This method is not called as part of a run, meaning you cannot rely on it being called!
    //
    // Set needs_rebuild to true if a rebuild is required after updating the configuration.
    // This is a heavy operation and should only be done if the outputs change.
    virtual void get_configuration([[maybe_unused]] Configuration& config,
                                   [[maybe_unused]] bool& needs_rebuild) {}

  public:
    const std::string name;
};

using NodeHandle = std::shared_ptr<Node>;

} // namespace merian_nodes
