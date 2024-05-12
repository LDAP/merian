#pragma once

#include "merian/utils/hash.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-nodes/graph/node_io.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include <merian/vk/sync/semaphore_binary.hpp>
#include <merian/vk/sync/semaphore_timeline.hpp>

#include <memory>
#include <optional>
#include <queue>
#include <unordered_set>

namespace merian {

// A unique object for each frame-in-flight.
class GraphFrameData {
    friend class Graph;

    uint64_t graph_version_identifier{0};
    std::unordered_map<NodeHandle, std::shared_ptr<Node::FrameData>> frame_data{};
};

// The result of the graph run.
// Nodes can insert semaphores that the user must submit together with the
// graph command buffer.
class GraphRun {
    friend class Graph;

  public:
    GraphRun(const Graph& graph, const std::shared_ptr<ExtensionVkDebugUtils> debug_utils)
        : graph(graph), debug_utils(debug_utils) {}

    void add_wait_semaphore(const BinarySemaphoreHandle& wait_semaphore,
                            const vk::PipelineStageFlags& wait_stage_flags) noexcept {
        wait_semaphores.push_back(*wait_semaphore);
        wait_stages.push_back(wait_stage_flags);
        wait_values.push_back(0);
    }

    void add_signal_semaphore(const BinarySemaphoreHandle& signal_semaphore) noexcept {
        signal_semaphores.push_back(*signal_semaphore);
        signal_values.push_back(0);
    }

    void add_wait_semaphore(const TimelineSemaphoreHandle& wait_semaphore,
                            const vk::PipelineStageFlags& wait_stage_flags,
                            const uint64_t value) noexcept {
        wait_semaphores.push_back(*wait_semaphore);
        wait_stages.push_back(wait_stage_flags);
        wait_values.push_back(value);
    }

    void add_signal_semaphore(const TimelineSemaphoreHandle& signal_semaphore,
                              const uint64_t value) noexcept {
        signal_semaphores.push_back(*signal_semaphore);
        signal_values.push_back(value);
    }

    void add_submit_callback(std::function<void(const QueueHandle& queue)> callback) noexcept {
        submit_callbacks.push_back(callback);
    }

    void request_rebuild() {
        rebuild_requested = true;
    }

    // increases with each run, resets at rebuild
    const uint64_t& get_iteration() const noexcept;

    // changes after every rebuild
    const uint64_t& get_graph_version_identifier() const noexcept;

    // Add this to the submit call for the graph command buffer
    const std::vector<vk::Semaphore>& get_wait_semaphores() const noexcept {
        return wait_semaphores;
    }

    // Add this to the submit call for the graph command buffer
    const std::vector<vk::PipelineStageFlags>& get_wait_stages() const noexcept {
        return wait_stages;
    }

    // Add this to the submit call for the graph command buffer
    const std::vector<vk::Semaphore>& get_signal_semaphores() const noexcept {
        return signal_semaphores;
    }

    // Add this to the submit call for the graph command buffer
    // The retuned pointer is valid until the next call to run.
    vk::TimelineSemaphoreSubmitInfo get_timeline_semaphore_submit_info() const noexcept {
        return vk::TimelineSemaphoreSubmitInfo{wait_values, signal_values};
    }

    // You must call every callback after you submited the graph command buffer
    // Or you use the execute_callbacks function.
    const std::vector<std::function<void(const QueueHandle& queue)>>
    get_submit_callbacks() const noexcept {
        return submit_callbacks;
    }

    // Call this after you submitted the graph command buffer
    void execute_callbacks(const QueueHandle& queue) const {
        for (auto& callback : submit_callbacks) {
            callback(queue);
        }
    }

    // Returns the profiler that is attached to this run.
    // Can be nullptr.
    const ProfilerHandle get_profiler() const {
        return profiler;
    }

  private:
    void reset(const ProfilerHandle profiler) {
        wait_semaphores.clear();
        wait_stages.clear();
        wait_values.clear();
        signal_semaphores.clear();
        signal_values.clear();
        submit_callbacks.clear();

        this->profiler = profiler;
        this->rebuild_requested = false;
    }

  private:
    const Graph& graph;
    std::vector<vk::Semaphore> wait_semaphores;
    std::vector<uint64_t> wait_values;
    std::vector<vk::PipelineStageFlags> wait_stages;
    std::vector<vk::Semaphore> signal_semaphores;
    std::vector<uint64_t> signal_values;

    std::vector<std::function<void(const QueueHandle& queue)>> submit_callbacks;
    ProfilerHandle profiler = nullptr;
    std::shared_ptr<ExtensionVkDebugUtils> debug_utils = nullptr;
    bool rebuild_requested = false;
    uint64_t graph_version_identifier = 0;
};

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
    friend GraphRun;

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

    struct NodeConnection {
        const NodeHandle dst;
        const std::string src_output;
        const std::string dst_input;

        bool operator==(const NodeConnection&) const = default;

        struct Hash {
            size_t operator()(const NodeConnection& c) const noexcept {
                std::size_t h = 0;
                hash_combine(h, c.dst, c.src_output, c.dst_input);
                return h;
            }
        };
    };

    struct NodeData {
        NodeHandle node;

        // A unique name for this node from the user. This is not node->name().
        // (on add)
        std::string name;

        // Cache inputs (node->describe_inputs())
        // (on add)
        std::vector<NodeInputDescriptorImage> image_input_descriptors{};
        std::vector<NodeInputDescriptorBuffer> buffer_input_descriptors{};

        // Set by the user using the public connect_* methods.
        // This information is used by connect_nodes to build the graph
        std::unordered_set<NodeConnection, NodeConnection::Hash> image_connections{};
        std::unordered_set<NodeConnection, NodeConnection::Hash> buffer_connections{};

        // For each input -> (Node, OutputIndex), e.g. to make sure every input is connected.
        // always has the size of the number of input descriptors.
        // (resized on add, filled on connect_nodes, reassigned with empty entries in reset)
        std::vector<std::tuple<NodeHandle, uint32_t>> image_input_connections{};
        std::vector<std::tuple<NodeHandle, uint32_t>> buffer_input_connections{};

        // For each output -> a list of inputs of (Node, InputIndex)
        // (on connect_nodes)
        std::vector<std::vector<std::tuple<NodeHandle, uint32_t>>> image_output_connections{};
        std::vector<std::vector<std::tuple<NodeHandle, uint32_t>>> buffer_output_connections{};

        // Cache outputs
        // (on compute_node_output_descriptors)
        std::vector<NodeOutputDescriptorImage> image_output_descriptors{};
        std::vector<NodeOutputDescriptorBuffer> buffer_output_descriptors{};

        // for each output -> (max_delay + 1) resources, accessed in iteration % (max_delay + 1).
        // (on allocate_outputs)
        std::vector<std::vector<std::shared_ptr<ImageResource>>> allocated_image_outputs{};
        std::vector<std::vector<std::shared_ptr<BufferResource>>> allocated_buffer_outputs{};

        // for each iteration -> For each output/input -> a list of resources which are
        // given to the node.
        // (on prepare_resource_sets)
        std::vector<NodeIO> precomputed_io{};

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

        // Keep here to prevent memory allocation
        Node::NodeStatus status{};

        uint32_t get_image_input_by_name(const std::string& name) {
            auto it =
                std::find_if(image_input_descriptors.begin(), image_input_descriptors.end(),
                             [&](NodeInputDescriptorImage& desc) { return desc.name == name; });

            if (it == image_input_descriptors.end())
                throw std::runtime_error{fmt::format("there is no image input '{}' on node {} ({})",
                                                     name, this->name, this->node->name())};

            return it - image_input_descriptors.begin();
        }

        uint32_t get_buffer_input_by_name(const std::string& name) {
            auto it =
                std::find_if(buffer_input_descriptors.begin(), buffer_input_descriptors.end(),
                             [&](NodeInputDescriptorBuffer& desc) { return desc.name == name; });

            if (it == buffer_input_descriptors.end())
                throw std::runtime_error{
                    fmt::format("there is no buffer input '{}' on node {} ({})", name, this->name,
                                this->node->name())};

            return it - buffer_input_descriptors.begin();
        }

        uint32_t get_image_output_by_name(const std::string& name) {
            auto it =
                std::find_if(image_output_descriptors.begin(), image_output_descriptors.end(),
                             [&](NodeOutputDescriptorImage& desc) { return desc.name == name; });

            if (it == image_output_descriptors.end())
                throw std::runtime_error{
                    fmt::format("there is no image output '{}' on node {} ({})", name, this->name,
                                this->node->name())};

            return it - image_output_descriptors.begin();
        }

        uint32_t get_buffer_output_by_name(const std::string& name) {
            auto it =
                std::find_if(buffer_output_descriptors.begin(), buffer_output_descriptors.end(),
                             [&](NodeOutputDescriptorBuffer& desc) { return desc.name == name; });

            if (it == buffer_output_descriptors.end())
                throw std::runtime_error{
                    fmt::format("there is no buffer output '{}' on node {} ({})", name, this->name,
                                this->node->name())};

            return it - buffer_output_descriptors.begin();
        }
    };

  public:
    // wait_queue: A queue we can wait for when rebuilding the graph (device.waitIdle() is used if
    // null).
    Graph(const SharedContext context,
          const ResourceAllocatorHandle allocator,
          const std::optional<QueueHandle> wait_queue = std::nullopt);

    // Add a node to the graph, returns the index of the node (can be used for connect and such).
    void add_node(const std::string& name, const std::shared_ptr<Node>& node);

    // Note: The connection is validated when the graph is build
    void connect_image(const NodeHandle& src,
                       const NodeHandle& dst,
                       const std::string& src_output,
                       const std::string& dst_input);

    // Note: The connection is validated when the graph is build
    void connect_buffer(const NodeHandle& src,
                        const NodeHandle& dst,
                        const std::string& src_output,
                        const std::string& dst_input);

    void request_rebuild() {
        rebuild_requested = true;
    }

    // Build the graph.
    // This is automatically called in cmd_run if a rebuild is requested
    // using request_rebuild() or internally by at least one node.
    void cmd_build(vk::CommandBuffer& cmd, const ProfilerHandle profiler = nullptr);

    // Runs the graph. On the first run, or if a rebuild is requested, the graph is automatically
    // built. frame_data must be unique for each frame-in-flight, reuse existing frame_data for best
    // performance since data might be reused.
    [[nodiscard]] const GraphRun& cmd_run(vk::CommandBuffer& cmd,
                                          GraphFrameData& graph_frame_data,
                                          const ProfilerHandle profiler = nullptr);

    // Collects configuration for the graph and all its nodes.
    void get_configuration(Configuration& config);

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;
    const std::optional<QueueHandle> wait_queue;
    const std::shared_ptr<ExtensionVkDebugUtils> debug_utils;
    GraphRun run;

    bool rebuild_requested = true;
    uint64_t current_iteration = 0;
    // changes at each rebuild
    uint64_t graph_version_identifier = 0;

    std::unordered_map<std::string, NodeHandle> node_from_name;
    std::unordered_map<NodeHandle, NodeData> node_data;

    // topological order of nodes
    std::vector<NodeHandle> flat_topology;

    // required in cmd_barrier_for_node, stored here to prevent memory allocation
    std::vector<vk::ImageMemoryBarrier2> image_barriers_for_set;
    std::vector<vk::BufferMemoryBarrier2> buffer_barriers_for_set;

    double duration_last_run;
    double duration_last_build;

  private: // Helpers
    // Throws if the graph is not fully connected.
    // Returns a topological order of the nodes.
    std::vector<NodeHandle> connect_nodes();

    // nodes without inputs or with delayed inputs only
    std::queue<NodeHandle> start_nodes();

    // Visites nodes in topological order as far as they are connected or a cycle is detected.
    // Returns the number of visited nodes.
    // Throws if the undelayed graph is not acyclic (feedback edges must have a delay of at
    // least 1).
    uint32_t topological_visit(const std::function<void(NodeHandle&, NodeData&)> visitor);

    // Calls node->describe_outputs with the appropriate parameters and populates the data object.
    // Requires that all inputs are connected.
    void compute_node_output_descriptors(NodeHandle& node, NodeData& data);

    void print_error_missing_inputs();

    std::string connections(NodeHandle& src);

    // Allocates the outputs for each node
    void allocate_outputs();

    // Depending on the delay the resources of a node changes on each iteration
    // the "resource sets" for these iterations are prepared here.
    void prepare_resource_sets();

    void cmd_build_node(vk::CommandBuffer& cmd, NodeHandle& node);

    // Insert the according barriers for that node
    void cmd_run_node(vk::CommandBuffer& cmd,
                      NodeHandle& node,
                      NodeData& data,
                      GraphFrameData& graph_frame_data);

    // Inserts the necessary barriers for a node and a set index
    void cmd_barrier_for_node(vk::CommandBuffer& cmd, NodeData& data, const uint32_t& set_idx);

    // Resets all data, so that the graph can be rebuild
    void reset_graph();

    void get_configuration_io_for_node(Configuration& config, NodeData& data);
};

} // namespace merian
