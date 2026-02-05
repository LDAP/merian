#pragma once

#include "connector_input.hpp"
#include "connector_output.hpp"
#include "graph_run.hpp"
#include "merian/utils/properties_json_dump.hpp"
#include "merian/utils/properties_json_load.hpp"
#include "node_io.hpp"

#include "merian/utils/properties.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"
#include "merian/vk/extension/extension.hpp"

#include <memory>

namespace merian {

class GraphInfo {};

class Node : public std::enable_shared_from_this<Node> {
  public:
    using NodeStatusFlags = uint32_t;

    enum NodeStatusFlagBits {
        // Ensures the graph is reconnected before the next call to process(...)
        NEEDS_RECONNECT = 0b1,
        // In on_connected: Resets frame data of EVERY frame in flight.
        // In pre_process: Resets frame data of the next run ONLY.
        RESET_IN_FLIGHT_DATA = 0b10,
        // removes the node from the graph
        REMOVE_NODE = 0b100,
    };

  public:
    Node() {}

    virtual ~Node() {}

    // -----------------------------------------------------------

    /**
     * @brief Request context extensions that this node requires.
     *
     * Called during graph initialization to determine which context extensions the node needs.
     * The graph will ensure these extensions are loaded from the registry before context creation.
     * Extensions can have dependencies on other extensions which will be resolved automatically.
     *
     * @return Vector of extension names to be loaded (e.g., {"glfw", "resources"})
     */
    virtual std::vector<std::string> request_context_extensions() {
        return {};
    }

    /**
     * @brief Query instance-level requirements.
     *
     * Called during context creation to determine what instance extensions and validation layers
     * this node requires. The node should check query_info.supported_extensions and
     * query_info.supported_layers to verify requirements are available.
     *
     * @param query_info Context containing supported extensions/layers and extension container
     * @return InstanceSupportInfo with supported flag and required extensions/layers
     */
    virtual InstanceSupportInfo
    query_instance_support(const InstanceSupportQueryInfo& /*query_info*/) {
        return InstanceSupportInfo{true};
    }

    /**
     * @brief Query if the node can run on a device and what it requires.
     *
     * Called during physical device selection to determine if this node is supported on the
     * device and what device extensions, features, and SPIR-V capabilities it requires.
     * If the node cannot run on the device, return supported=false.
     *
     * @param query_info Context containing physical device, queue info, and extension container
     * @return DeviceSupportInfo with supported flag and required extensions/features/SPIR-V
     */
    virtual DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& /*query_info*/) {
        return DeviceSupportInfo{true};
    }

    // Initialize for this context (and device), now knowing which (physical) device to use.
    // The graph must ensure that you get a device from a physical device for which
    // query_device_support returned true and all requirements are enabled.
    //
    // Use the allocator to allocate static data, that does not depend on graph configuration.
    virtual void initialize(const ContextHandle& /*context*/,
                            const ResourceAllocatorHandle& /*allocator*/) {}

    // This might be called at any time of the graph lifecycle. Must be consistent with dump_config.
    virtual NodeStatusFlags load_config(const nlohmann::json& json);

    // This might be called at any time of the graph lifecycle. Must be consistent with load_config.
    virtual nlohmann::json dump_config();

    // -----------------------------------------------------------

    // Called each time the graph attempts to connect nodes.
    // If you need to access the resources directly, you need to maintain a copy of the InputHandle.
    //
    // Note that input and output names must be unique.
    //
    // If you throw node_error or compilation_failed the graph will disable the node for this
    // connect attempt and set the error state for this node. Your inputs will then be invisible.
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
    //
    // If you throw node_error or compilation_failed the graph will disable the node for this
    // connect attempt and set the error state for this node.
    [[nodiscard]]
    virtual std::vector<OutputConnectorHandle>
    describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
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
    //
    // Here also delayed inputs can be accessed from io_layout.
    [[nodiscard]]
    virtual NodeStatusFlags
    on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                 [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {
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
    virtual NodeStatusFlags pre_process([[maybe_unused]] const GraphRun& run,
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
    //
    // You can throw node_error and compilation_failed here. The graph then attempts to finish the
    // run and rebuild, however this is not supported and not recommended.
    virtual void process([[maybe_unused]] GraphRun& run,
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
    virtual NodeStatusFlags properties([[maybe_unused]] Properties& props) {
        return {};
    }
};

using NodeHandle = std::shared_ptr<Node>;

} // namespace merian
