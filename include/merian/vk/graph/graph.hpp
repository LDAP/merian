#pragma once

#include "merian/vk/graph/node.hpp"
#include "merian/vk/graph/node_io.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/math.hpp"

#include <memory>
#include <queue>
#include <unordered_set>
#include <variant>

namespace merian {

/**
 * @brief      This class describes a general processing graph.
 *
 * Nodes can define their required inputs and outputs.
 * The graph wires up the nodes and allocates the memory for outputs.
 * Memory may be aliased if 'persistent=false' for an output.
 * The graph can also buffer resources is delay > 0.
 *
 * Note that it is not possible to access the same output twice from the same node
 * with equal value for delay. Since the graph does also insert memory barriers and
 * does layout transitions it is not possible to access
 *
 * These barriers are automatically inserted:
 * - For buffers and images: Before they are used as input or output
 *   for an output the access flags are set to the exact flags of that output
 *   for an input the access flags are set to the disjunction of all access flags of all inputs that
 *   use this resource.
 * - For images: Whenever a layout transition is required
 *
 */
class Graph : public std::enable_shared_from_this<Graph> {

  private:
    // Holds information about images that were allocated by this graph
    struct ImageResource {
        ImageHandle image;

        // for barrier insertions
        vk::PipelineStageFlags2 current_stage_flags;
        vk::AccessFlags2 current_access_flags;

        // to detect if a barrier is needed
        bool last_used_as_output = false;

        // combined pipeline stage flags of all inputs
        vk::PipelineStageFlags2 input_stage_flags;
        // combined access flags of all inputs
        vk::AccessFlags2 input_access_flags;
    };

    // Holds information about buffers that were allocated by this graph
    struct BufferResource {
        BufferHandle buffer;

        // for barrier insertions
        vk::PipelineStageFlags2 current_stage_flags;
        vk::AccessFlags2 current_access_flags;

        // to detect which src flags are needed
        // if true: Use the access and pipeline flags from the output
        // if false: use the input_*_flags
        bool last_used_as_output = false;

        // combined pipeline stage flags of all inputs
        vk::PipelineStageFlags2 input_stage_flags;
        // combined access flags of all inputs
        vk::AccessFlags2 input_access_flags;
    };

    struct NodeData {
        NodeHandle node;

        // A name for this node from the user. This is not node->name().
        std::string name;

        // Cache inputs (on add)
        std::vector<NodeInputDescriptorImage> image_input_descriptors{};
        std::vector<NodeInputDescriptorBuffer> buffer_input_descriptors{};

        // For each input -> (Node, OutputIndex), e.g. to make sure every input is connected.
        // (resize on add, insert on connect)
        std::vector<std::tuple<NodeHandle, uint32_t>> image_input_connections{};
        std::vector<std::tuple<NodeHandle, uint32_t>> buffer_input_connections{};

        // For each output -> a list of inputs of (Node, InputIndex)
        // on connect
        std::vector<std::vector<std::tuple<NodeHandle, uint32_t>>> image_output_connections{};
        std::vector<std::vector<std::tuple<NodeHandle, uint32_t>>> buffer_output_connections{};

        // Cache outputs (on calculate_outputs)
        std::vector<NodeOutputDescriptorImage> image_output_descriptors{};
        std::vector<NodeOutputDescriptorBuffer> buffer_output_descriptors{};

        // for each output -> (max_delay + 1) resources, accessed in iteration % (max_delay + 1).
        // (on allocate_outputs)
        std::vector<std::vector<std::shared_ptr<ImageResource>>> allocated_image_outputs{};
        std::vector<std::vector<std::shared_ptr<BufferResource>>> allocated_buffer_outputs{};

        // for each iteration -> For each output/input -> a list of resources which are
        // given to the node.
        // (on prepare_resource_sets)
        std::vector<std::vector<ImageHandle>> precomputed_input_images{};
        std::vector<std::vector<BufferHandle>> precomputed_input_buffers{};
        std::vector<std::vector<ImageHandle>> precomputed_output_images{};
        std::vector<std::vector<BufferHandle>> precomputed_output_buffers{};

        // as precomputed_input_images but hold a reference to resource
        // needed for barrier insertion
        std::vector<std::vector<std::shared_ptr<ImageResource>>>
            precomputed_input_images_resource{};
        std::vector<std::vector<std::shared_ptr<BufferResource>>>
            precomputed_input_buffers_resource{};
        std::vector<std::vector<std::shared_ptr<ImageResource>>>
            precomputed_output_images_resource{};
        std::vector<std::vector<std::shared_ptr<BufferResource>>>
            precomputed_output_buffers_resource{};
    };

  public:
    // wait_queue: A queue we can wait for when rebuilding the graph (device.waitIdle() is used if
    // null).
    Graph(const SharedContext context,
          const ResourceAllocatorHandle allocator,
          const std::optional<QueueHandle> wait_queue = std::nullopt);

    // Add a node to the graph, returns the index of the node (can be used for connect and such).
    void add_node(const std::string name, const std::shared_ptr<Node>& node);

    // Note: The connection is validated when the graph is build
    void connect_image(const NodeHandle& src,
                       const NodeHandle& dst,
                       const uint32_t src_output,
                       const uint32_t dst_input);

    // Note: The connection is validated when the graph is build
    void connect_buffer(const NodeHandle& src,
                        const NodeHandle& dst,
                        const uint32_t src_output,
                        const uint32_t dst_input);

    void request_rebuild() {
        rebuild_requested = true;
    }

    // Runs the graph. On the first run or if rebuild is requested the graph is build.
    void cmd_run(vk::CommandBuffer& cmd);

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;
    const std::optional<QueueHandle> wait_queue;

    bool rebuild_requested = true;
    uint64_t current_iteration = 0;

    std::unordered_map<std::string, NodeHandle> node_from_name;
    std::unordered_map<NodeHandle, NodeData> node_data;

    // topological order of nodes
    std::vector<NodeHandle> flat_topology;

    // required in cmd_barrier_for_node, stored here to prevent memory allocation
    std::vector<vk::ImageMemoryBarrier2> image_barriers_for_set;
    std::vector<vk::BufferMemoryBarrier2> buffer_barriers_for_set;

  private: // Helpers
    // Makes sure every input is connected
    void validate_inputs();

    // nodes without inputs or with delayed inputs only
    std::queue<NodeHandle> start_nodes();

    // For each node input find the corresponding output descriptors (image_outputs_descriptors,
    // buffer_outputs_descriptors). Inserts subsequent nodes to the queue if all inputs are
    // satisfied.
    void calculate_outputs(NodeHandle& node,
                           std::unordered_set<NodeHandle>& visited,
                           std::queue<NodeHandle>& queue);

    void log_connections(NodeHandle& src);

    // Allocates the outputs for each node
    void allocate_outputs();

    // Depending on the delay the resources of a node changes on each iteration
    // the "resource sets" for these iterations are prepared here.
    void prepare_resource_sets();

    void cmd_build(vk::CommandBuffer& cmd);

    void cmd_build_node(vk::CommandBuffer& cmd, NodeHandle& node);

    // Insert the according barriers for that node
    void cmd_run_node(vk::CommandBuffer& cmd, NodeHandle& node);

    // Inserts the necessary barriers for a node and a set index
    void cmd_barrier_for_node(vk::CommandBuffer& cmd, NodeData& data, uint32_t& set_idx);

    // Resets all data, so that the graph can be rebuild
    void reset_graph();
};

} // namespace merian
