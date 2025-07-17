#pragma once

#include "merian-nodes/graph/graph_description.hpp"
#include "merian-nodes/graph/node.hpp"

namespace merian_nodes {

/* A built graph, that is ready to be executed.
 *
 * Implements Node interface for containerization.
 */
class Graph : public Node {

  public:
    /**
     * @brief      An empty graph.
     */
    Graph() {}

    // Removes all nodes and connections from the graph.
    void clear() {}

    /**
     * @brief      Builds a graph from a description.
     *
     * @param[in]  description  The graph structure and config.
     * @param[in]  auto_repair  Updates the description, for example if illegal connections are
     * detected
     *
     * @throws invalid_connection if there is an illegal connection present in the description and
     * auto_repair is false
     * @throws connection_missing if there is an connection missing and allow_partial_build is false
     * @throws build_error for everything else that prevents a graph build
     */
    void build(GraphDescription& description, const bool auto_repair = true) {}

  public:
    // All unconnected node inputs
    std::vector<InputConnectorHandle> describe_inputs() override {
        return {};
    }

    // All outputs that are marked as graph_output
    std::vector<OutputConnectorHandle>
    describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) override {
        return {};
    }

    NodeStatusFlags
    on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                 [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) override {
        return {};
    }

    NodeStatusFlags pre_process([[maybe_unused]] const GraphRun& run,
                                [[maybe_unused]] const NodeIO& io) override {
        return {};
    }

    void process([[maybe_unused]] GraphRun& run,
                 [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                 [[maybe_unused]] const NodeIO& io) override {}

    NodeStatusFlags properties([[maybe_unused]] Properties& config) override {
        return {};
    }

    nlohmann::json dump_config() override {
        merian::JSONDumpProperties props;
        std::ignore = properties(props);
        return props.get();
    }

    //
    void load_config(const nlohmann::json& json) override {
        merian::JSONLoadProperties props(json);
    }

  private:
    // Means all non-optional inputs of the nodes are connected and
    // the graph can be executed by a graph runtime. If this is false, the graph can still be
    // composed in a bigger graph.
    bool runnable{true};

    struct NodeData {};

    // the layers of the graph. Each layer is independent of the layer before.
    std::vector<std::vector<NodeData>> graph;
};

using GraphHandle = std::shared_ptr<Graph>;

} // namespace merian_nodes
