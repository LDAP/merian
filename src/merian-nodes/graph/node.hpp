#pragma once

#include "connector_input.hpp"
#include "connector_output.hpp"
#include "graph_run.hpp"
#include "node_io.hpp"

#include "merian/utils/properties.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include <memory>

namespace merian_nodes {

class Node : public std::enable_shared_from_this<Node> {
  public:
    using NodeStatusFlags = uint32_t;

    enum NodeStatusFlagBits {
        // Ensures the graph is reconnected before the next call to process(...)
        NEEDS_RECONNECT = 0b1,
        // In on_connected: Resets frame data of EVERY frame in flight.
        // In pre_process: Resets frame data of the next run ONLY.
        RESET_IN_FLIGHT_DATA = 0b10,
    };

  public:
    Node(const std::string& name) : name(name) {}

    virtual ~Node() {}

    // Called each time the graph attempts to connect nodes.
    // If you need to access the resources directly, you need to maintain a copy of the InputHandle.
    //
    // Note that input and output names must be unique.
    [[nodiscard]]
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
    [[nodiscard]]
    virtual std::vector<OutputConnectorHandle>
    describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
        return {};
    }

    // Called when the graph is fully connected and all inputs and outputs are defined.
    // This is a good place to create layouts and pipelines.
    // This might be called multiple times in the nodes life-cycle (whenever a connection changes or
    // other nodes request reconnects). It can be assumed that at the time of calling processing of
    // all in-flight data has finished, that means old pipelines and such can be safely destroyed.
    //
    // The descriptor set layout is automatically constructed from the inputs and outputs.
    // It contains all input and output connectors for which get_descriptor_info() method does not
    // return std::nullopt. The order is guaranteed to be all inputs in the order of
    // describe_inputs() then outputs in the order of describe_outputs().
    [[nodiscard]]
    virtual NodeStatusFlags
    on_connected([[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {
        return {};
    }

    // Called before each run.
    //
    // Note that requesting a reconnect is a heavy operation and should only be called if the
    // outputs change. The graph then has to reconnect itself before calling cmd_process. Note, that
    // this method is called again after the reconnect until no node requests a reconnect.
    //
    // Here you can access the resources for the run or set your own, depending on the descriptor
    // type. It is guaranteed that the descriptor set in process(...) is accordingly updated. If you
    // update resources in process(...) the descriptor set will reflect the changes one iteration
    // later.
    [[nodiscard]]
    virtual NodeStatusFlags pre_process([[maybe_unused]] GraphRun& run,
                                        [[maybe_unused]] const NodeIO& io) {
        return {};
    }

    // Do your main GPU processing here.
    //
    // You do not need to insert barriers for node inputs and outputs if not stated otherwise in the
    // connector documentation. If you need to perform layout transitions use the barrier() methods
    // of the images.
    //
    // You can provide data that that is required for the current run by setting the io map
    // in_flight_data. The pointer is persisted and supplied again after (graph ring size - 1) runs.
    virtual void process([[maybe_unused]] GraphRun& run,
                         [[maybe_unused]] const vk::CommandBuffer& cmd,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         [[maybe_unused]] const NodeIO& io) {}

    // Declare your configuration options and output status information.
    // This method is not called as part of a run, meaning you cannot rely on it being called!
    //
    // Return NEEDS_RECONNECT if reconnecting is required after updating the configuration.
    // This is a heavy operation and should only be done if the outputs change.
    //
    // Normally this method is called by the graph configuration(), if you want to call it directly
    // you need to handle the NodeStatusFlags accordingly.
    [[nodiscard]]
    virtual NodeStatusFlags properties([[maybe_unused]] Properties& config) {
        return {};
    }

  public:
    const std::string name;
};

using NodeHandle = std::shared_ptr<Node>;

} // namespace merian_nodes
