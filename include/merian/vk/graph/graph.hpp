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

class Graph : public std::enable_shared_from_this<Graph> {

  private:
    struct ImageResource {
        ImageHandle image;
        vk::PipelineStageFlags2 current_stage_flags;
        vk::AccessFlags2 current_access_flags;
    };

    struct BufferResource {
        BufferHandle buffer;
        vk::PipelineStageFlags2 current_stage_flags;
        vk::AccessFlags2 current_access_flags;
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
        std::vector<NodeOutputDescriptorImage> image_outputs_descriptors{};
        std::vector<NodeOutputDescriptorBuffer> buffer_outputs_descriptors{};

        // for each output -> (max_delay + 1) resources, accessed in iteration % (max_delay + 1).
        // (on allocate_outputs)
        std::vector<std::vector<ImageResource>> output_images{};
        std::vector<std::vector<BufferResource>> output_buffers{};

        // for each iteration (max_delay + 1) -> For each output -> a list of resources which are
        // given to the node.
        // (on prepare_resource_sets)
        std::vector<std::vector<ImageHandle>> precomputed_input_images{};
        std::vector<std::vector<BufferHandle>> precomputed_input_buffers{};
        std::vector<std::vector<ImageHandle>> precomputed_output_images{};
        std::vector<std::vector<BufferHandle>> precomputed_output_buffers{};

        // // for each iteration (max_delay + 1) -> For each input -> a memory barrier
        // // (on allocate_outputs)
        // std::vector<std::vector<vk::ImageMemoryBarrier2>> input_images_barriers{};
        // // for each iteration (max_delay + 1) -> For each output -> a memory barrier
        // // (on allocate_outputs)
        // std::vector<std::vector<vk::ImageMemoryBarrier2>> output_images_barriers{};

        // // one combined memory barrier for all buffers
        // vk::MemoryBarrier2 memory_barrier{};
    };

  public:
    Graph(const SharedContext context, const ResourceAllocatorHandle allocator)
        : context(context), allocator(allocator) {}

    // Add a node to the graph, returns the index of the node (can be used for connect and such).
    void add_node(const std::string name, const std::shared_ptr<Node>& node) {
        if (node_from_name.contains(name)) {
            throw std::invalid_argument{
                fmt::format("graph already contains a node with name '{}'", name)};
        }
        if (node_data.contains(node)) {
            throw std::invalid_argument{
                fmt::format("graph already contains this node with a different name '{}'", name)};
        }

        auto [image_inputs, buffer_inputs] = node->describe_inputs();
        node_from_name[name] = node;
        node_data[node] = {node, name, image_inputs, buffer_inputs};
        node_data[node].image_input_connections.resize(image_inputs.size());
        node_data[node].buffer_input_connections.resize(buffer_inputs.size());
    }

    // Note: The connection is validated when the graph is build
    void connect_image(const NodeHandle& src,
                       const NodeHandle& dst,
                       const uint32_t src_output,
                       const uint32_t dst_input) {
        if (src_output >= node_data[src].image_output_connections.size()) {
            node_data[src].image_output_connections.resize(src_output + 1);
        }
        // dst_input is valid
        if (dst_input >= node_data[dst].image_input_connections.size()) {
            throw std::invalid_argument{
                fmt::format("There is no input '{}' on node '{}'", dst_input, node_data[dst].name)};
        }
        if (std::get<0>(node_data[dst].image_input_connections[dst_input])) {
            throw std::invalid_argument{
                fmt::format("The input '{}' on node '{}' is already connected", dst_input,
                            node_data[dst].name)};
        }
        node_data[dst].image_input_connections[dst_input] = {src, src_output};

        // make sure the same underlying resource is not accessed twice:
        for (auto& [n, i] : node_data[src].image_output_connections[src_output]) {
            if (n == dst && node_data[dst].image_input_descriptors[i].delay ==
                                node_data[dst].image_input_descriptors[dst_input].delay) {
                throw std::invalid_argument{fmt::format(
                    "You are trying to access the same underlying image of node '{}' twice from "
                    "node '{}' with connections {} -> {}, {} -> {}: ",
                    node_data[src].name, node_data[dst].name, src_output, i, src_output,
                    dst_input)};
            }
        }
        node_data[src].image_output_connections[src_output].emplace_back(dst, dst_input);
    }

    // Note: The connection is validated when the graph is build
    void connect_buffer(const NodeHandle& src,
                        const NodeHandle& dst,
                        const uint32_t src_output,
                        const uint32_t dst_input) {
        if (src_output >= node_data[src].buffer_output_connections.size()) {
            node_data[src].buffer_output_connections.resize(src_output + 1);
        }
        // dst_input is valid
        assert(dst_input < node_data[dst].buffer_input_connections.size());
        // nothing is connected to this input
        assert(!std::get<0>(node_data[dst].buffer_input_connections[dst_input]));
        node_data[dst].buffer_input_connections[dst_input] = {src, src_output};

        // make sure the same underlying resource is not accessed twice:
        for (auto& [n, i] : node_data[src].buffer_output_connections[src_output]) {
            if (n == dst && node_data[dst].buffer_input_descriptors[i].delay ==
                                node_data[dst].buffer_input_descriptors[dst_input].delay) {
                throw std::invalid_argument{fmt::format(
                    "You are trying to access the same underlying buffer of node '{}' twice from "
                    "node '{}' with connections {} -> {}, {} -> {}: ",
                    node_data[src].name, node_data[dst].name, src_output, i, src_output,
                    dst_input)};
            }
        }
        node_data[src].buffer_output_connections[src_output].emplace_back(dst, dst_input);
    }

    void cmd_build(vk::CommandBuffer& cmd) {
        if (node_data.empty())
            return;

        validate_inputs();

        // Visit nodes in topological order
        // to calculate outputs, barriers and such.
        // Feedback edges must have a delay of at least 1.
        flat_topology.resize(node_data.size());
        std::unordered_set<NodeHandle> visited;
        std::queue<NodeHandle> queue = start_nodes();

        uint32_t node_index = 0;
        while (!queue.empty()) {
            flat_topology[node_index] = queue.front();
            queue.pop();

            topology_index[flat_topology[node_index]] = node_index;
            visited.insert(flat_topology[node_index]);
            calculate_outputs(flat_topology[node_index], visited, queue);
            log_connections(flat_topology[node_index]);

            node_index++;
        }
        // For some reason a node was not appended to the queue
        assert(node_index == node_data.size());
        allocate_outputs();
        prepare_resource_sets();

        for (auto& node : flat_topology) {
            NodeData& data = node_data[node];
            // TODO: See cmd_run
            node->cmd_build(cmd, data.precomputed_input_images, data.precomputed_input_buffers,
                            data.precomputed_output_images, data.precomputed_output_buffers);
        }

        graph_built = true;
        current_iteration = 0;
    }

    void cmd_run(vk::CommandBuffer& cmd) {
        assert(graph_built);

        // for (auto& node : flat_topology) {
        //     NodeData& data = node_data[node];
        //     uint32_t set_index = current_iteration % data.input_images.size();
        //     auto& in_images = data.input_images[set_index];
        //     auto& in_buffers = data.input_buffers[set_index];
        //     auto& out_images = data.output_images[set_index];
        //     auto& out_buffers = data.output_buffers[set_index];

        //     node->cmd_process(cmd, current_iteration, set_index, in_images, in_buffers,
        //     out_images,
        //                       out_buffers);

        //     // TODO:
        //     // - Serialize layers too to determine image transisions and such -> On output might
        //     be
        //     // read
        //     //   by multiple nodes as input, but they want different layouts!
        //     // - Insert current layout into image barriers!
        //     // vk::DependencyInfo dep_info{
        //     //     {}, data.memory_barrier, {}, data.images_barriers[set_index]};
        //     // cmd.pipelineBarrier2(dep_info);
        // }

        current_iteration++;
    }

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;

    bool graph_built = false;
    uint64_t current_iteration = 0;

    std::unordered_map<std::string, NodeHandle> node_from_name;
    std::unordered_map<NodeHandle, NodeData> node_data;

    // topological order of nodes
    std::vector<NodeHandle> flat_topology;
    std::unordered_map<NodeHandle, uint32_t> topology_index;

    // Contains all nodes, barriers, ... in order to render frames
    // std::vector<TimelineElement>

  private: // Helpers
    // Makes sure every input is connected
    void validate_inputs() {
        for (auto& [dst_node, dst_data] : node_data) {
            // Images
            for (uint32_t i = 0; i < dst_data.image_input_descriptors.size(); i++) {
                auto& [src_node, src_connection_idx] = dst_data.image_input_connections[i];
                auto& in_desc = dst_data.image_input_descriptors[i];
                if (src_node == nullptr) {
                    throw std::runtime_error{
                        fmt::format("image input '{}' ({}) of node '{}' was not connected!",
                                    in_desc.name, i, dst_data.name)};
                }
                if (src_node == dst_node && in_desc.delay == 0) {
                    throw std::runtime_error{fmt::format(
                        "node '{}'' is connected to itself with delay 0, maybe you want "
                        "to use a persistent output?",
                        dst_data.name)};
                }
            }
            // Buffers
            for (uint32_t i = 0; i < dst_data.buffer_input_descriptors.size(); i++) {
                auto& [src_node, src_connection_idx] = dst_data.buffer_input_connections[i];
                auto& in_desc = dst_data.buffer_input_descriptors[i];
                if (src_node == nullptr) {
                    throw std::runtime_error{
                        fmt::format("buffer input {} ({}) of node {} was not connected!",
                                    in_desc.name, i, dst_data.name)};
                }
                if (src_node == dst_node && in_desc.delay == 0) {
                    throw std::runtime_error{
                        fmt::format("node {} is connected to itself with delay 0, maybe you want "
                                    "to use a persistent output?",
                                    dst_data.name)};
                }
            }
        }
    }

    // nodes without inputs or with delayed inputs only
    std::queue<NodeHandle> start_nodes() {
        std::queue<NodeHandle> queue;

        // Find nodes without inputs or with delayed inputs only
        for (auto& [node, node_data] : node_data) {
            if (node_data.image_input_descriptors.empty() &&
                node_data.buffer_input_descriptors.empty()) {
                queue.push(node);
                continue;
            }
            uint32_t num_non_delayed = 0;
            for (auto& desc : node_data.image_input_descriptors) {
                if (desc.delay == 0)
                    num_non_delayed++;
            }
            for (auto& desc : node_data.buffer_input_descriptors) {
                if (desc.delay == 0)
                    num_non_delayed++;
            }

            if (num_non_delayed == 0)
                queue.push(node);
        }

        return queue;
    }

    // For each node input find the corresponding output descriptors (image_outputs_descriptors,
    // buffer_outputs_descriptors). Inserts subsequent nodes to the queue if all inputs are
    // satisfied.
    void calculate_outputs(NodeHandle& node,
                           std::unordered_set<NodeHandle>& visited,
                           std::queue<NodeHandle>& queue) {
        NodeData& data = node_data[node];

        std::vector<NodeOutputDescriptorImage> connected_image_outputs;
        std::vector<NodeOutputDescriptorBuffer> connected_buffer_outputs;

        // find outputs that are connected to inputs.
        for (uint32_t i = 0; i < data.image_input_descriptors.size(); i++) {
            auto& [src_node, src_output_idx] = data.image_input_connections[i];
            auto& in_desc = data.image_input_descriptors[i];
            if (in_desc.delay > 0)
                connected_image_outputs.push_back(Node::FEEDBACK_OUTPUT_IMAGE);
            else
                connected_image_outputs.push_back(
                    node_data[src_node].image_outputs_descriptors[src_output_idx]);
        }
        for (uint32_t i = 0; i < data.buffer_input_descriptors.size(); i++) {
            auto& [src_node, src_output_idx] = data.buffer_input_connections[i];
            auto& in_desc = data.buffer_input_descriptors[i];
            if (in_desc.delay > 0)
                connected_buffer_outputs.push_back(Node::FEEDBACK_OUTPUT_BUFFER);
            else
                connected_buffer_outputs.push_back(
                    node_data[src_node].buffer_outputs_descriptors[src_output_idx]);
        }

        // get outputs from node
        std::tie(data.image_outputs_descriptors, data.buffer_outputs_descriptors) =
            node->describe_outputs(connected_image_outputs, connected_buffer_outputs);

        // validate that the user did not try to connect something from an non existent output,
        // since on connect we did not know the number of output descriptors
        if (data.image_output_connections.size() > data.image_outputs_descriptors.size()) {
            throw std::runtime_error{fmt::format("image output index '{}' is invalid for node '{}'",
                                                 data.image_output_connections.size() - 1,
                                                 data.name)};
        }
        if (data.buffer_output_connections.size() > data.buffer_outputs_descriptors.size()) {
            throw std::runtime_error{
                fmt::format("buffer output index '{}' is invalid for node '{}'",
                            data.buffer_output_connections.size() - 1, data.name)};
        }
        data.image_output_connections.resize(data.image_outputs_descriptors.size());
        data.buffer_output_connections.resize(data.buffer_outputs_descriptors.size());

        // check for all subsequent nodes if we visited all "requirements" and add to queue.
        // also, fail if we see a node again! (in both cases exclude "feedback" edges)

        // find all subsequent nodes that are connected over a edge with delay = 0.
        std::unordered_set<NodeHandle> candidates;
        for (auto& output : data.image_output_connections) {
            for (auto& [dst_node, image_input_idx] : output) {
                if (node_data[dst_node].image_input_descriptors[image_input_idx].delay == 0) {
                    candidates.insert(dst_node);
                }
            }
        }
        for (auto& output : data.buffer_output_connections) {
            for (auto& [dst_node, buffer_input_idx] : output) {
                if (node_data[dst_node].buffer_input_descriptors[buffer_input_idx].delay == 0) {
                    candidates.insert(dst_node);
                }
            }
        }

        // add to queue if all "inputs" were visited
        for (const NodeHandle& candidate : candidates) {
            if (visited.contains(candidate)) {
                // Back-edges with delay > 1 are allowed!
                throw std::runtime_error{
                    fmt::format("undelayed (edges with delay = 0) graph is not acyclic! {} -> {}",
                                data.name, node_data[candidate].name)};
            }
            bool satisfied = true;
            NodeData& candidate_data = node_data[candidate];
            for (auto& [src_node, src_output_idx] : candidate_data.image_input_connections) {
                satisfied &= visited.contains(src_node);
            }
            for (auto& [src_node, src_output_idx] : candidate_data.buffer_input_connections) {
                satisfied &= visited.contains(src_node);
            }
            if (satisfied) {
                queue.push(candidate);
            }
        }
    }

    void log_connections(NodeHandle& src) {
#ifdef NDEBUG
        return;
#endif

        NodeData& src_data = node_data[src];
        for (uint32_t i = 0; i < src_data.image_outputs_descriptors.size(); i++) {
            auto& src_out_desc = src_data.image_outputs_descriptors[i];
            auto& src_output = src_data.image_output_connections[i];
            for (auto& [dst_node, image_input_idx] : src_output) {
                NodeData& dst_data = node_data[dst_node];
                auto& dst_in_desc = dst_data.image_input_descriptors[image_input_idx];
                SPDLOG_DEBUG("image connection: {}({}) --{}-> {}({})", src_data.name,
                             src_out_desc.name, dst_in_desc.delay, dst_data.name, dst_in_desc.name);
            }
        }
        for (uint32_t i = 0; i < src_data.buffer_outputs_descriptors.size(); i++) {
            auto& src_out_desc = src_data.buffer_outputs_descriptors[i];
            auto& src_output = src_data.buffer_output_connections[i];
            for (auto& [dst_node, buffer_input_idx] : src_output) {
                NodeData& dst_data = node_data[dst_node];
                auto& dst_in_desc = dst_data.buffer_input_descriptors[buffer_input_idx];
                SPDLOG_DEBUG("buffer connection: {}({}) --{}-> {}({})", src_data.name,
                             src_out_desc.name, dst_in_desc.delay, dst_data.name, dst_in_desc.name);
            }
        }
    }

    // Allocates the outputs for each node
    void allocate_outputs() {
        for (auto& [src_node, src_data] : node_data) {
            // Buffers
            src_data.output_buffers.resize(src_data.buffer_outputs_descriptors.size());
            for (uint32_t src_out_idx = 0; src_out_idx < src_data.buffer_outputs_descriptors.size();
                 src_out_idx++) {
                auto& out_desc = src_data.buffer_outputs_descriptors[src_out_idx];
                vk::BufferUsageFlags usage_flags = out_desc.create_info.usage;
                uint32_t max_delay = 0;
                for (auto& [dst_node, dst_input_idx] :
                     src_data.buffer_output_connections[src_out_idx]) {
                    auto& in_desc = node_data[dst_node].buffer_input_descriptors[dst_input_idx];
                    max_delay = std::max(max_delay, in_desc.delay);
                    usage_flags |= in_desc.usage_flags;
                }
                // Create max_delay + 1 buffers
                for (uint32_t j = 0; j < max_delay + 1; j++) {
                    BufferHandle buffer = allocator->createBuffer(
                        out_desc.create_info.size, usage_flags, NONE,
                        fmt::format("node '{}' buffer, output '{}', copy '{}'", src_data.name,
                                    out_desc.name, j));
                    src_data.output_buffers[src_out_idx].emplace_back(
                        buffer, vk::PipelineStageFlagBits2::eTopOfPipe, vk::AccessFlags2());
                }
            }

            // Images
            src_data.output_images.resize(src_data.image_outputs_descriptors.size());
            for (uint32_t src_out_idx = 0; src_out_idx < src_data.image_outputs_descriptors.size();
                 src_out_idx++) {
                auto& out_desc = src_data.image_outputs_descriptors[src_out_idx];
                vk::ImageCreateInfo create_info = out_desc.create_info;
                uint32_t max_delay = 0;
                for (auto& [dst_node, dst_input_idx] :
                     src_data.image_output_connections[src_out_idx]) {
                    auto& in_desc = node_data[dst_node].image_input_descriptors[dst_input_idx];
                    max_delay = std::max(max_delay, in_desc.delay);
                    create_info.usage |= in_desc.usage_flags;
                }
                // Create max_delay + 1 images
                for (uint32_t j = 0; j < max_delay + 1; j++) {
                    ImageHandle image = allocator->createImage(
                        create_info, NONE,
                        fmt::format("node '{}' image, output '{}', copy '{}'", src_data.name,
                                    out_desc.name, j));
                    src_data.output_images[src_out_idx].emplace_back(
                        image, vk::PipelineStageFlagBits2::eTopOfPipe, vk::AccessFlags2());
                }
            }
        }
    }

    // Depending on the delay the resources of a node changes on each iteration
    // the "resource sets" for these iterations are prepared here.
    void prepare_resource_sets() {
        for (auto& [dst_node, dst_data] : node_data) {
            // Find the lowest number of sets needed (lcm)
            std::vector<uint32_t> num_resources;

            // By checking how many copies of that resource exists in the sources
            for (auto& [src_node, src_output_idx] : dst_data.image_input_connections) {
                num_resources.push_back(node_data[src_node].output_images[src_output_idx].size());
            }
            for (auto& [src_node, src_output_idx] : dst_data.buffer_input_connections) {
                num_resources.push_back(node_data[src_node].output_buffers[src_output_idx].size());
            }
            // ...and how many output resources the node has
            for (auto& images : dst_data.output_images) {
                num_resources.push_back(images.size());
            }
            for (auto& buffers : dst_data.output_buffers) {
                num_resources.push_back(buffers.size());
            }

            // After this many iterations we can again use the first resource set
            uint32_t num_sets = lcm(num_resources);

            // Precompute resource sets for each iteration
            dst_data.precomputed_input_images.resize(num_sets);
            dst_data.precomputed_input_buffers.resize(num_sets);
            dst_data.precomputed_output_images.resize(num_sets);
            dst_data.precomputed_output_buffers.resize(num_sets);

            for (uint32_t set_idx = 0; set_idx < num_sets; set_idx++) {
                // Precompute inputs
                for (uint32_t i = 0; i < dst_data.image_input_descriptors.size(); i++) {
                    auto& [src_node, src_output_idx] = dst_data.image_input_connections[i];
                    auto& in_desc = dst_data.image_input_descriptors[i];
                    const uint32_t num_resources =
                        node_data[src_node].output_images[src_output_idx].size();
                    const uint32_t resource_idx =
                        (set_idx + num_resources - in_desc.delay) % num_resources;
                    const auto& resource =
                        node_data[src_node].output_images[src_output_idx][resource_idx];
                    dst_data.precomputed_input_images[set_idx].push_back(resource.image);
                }
                for (uint32_t i = 0; i < dst_data.buffer_input_descriptors.size(); i++) {
                    auto& [src_node, src_output_idx] = dst_data.buffer_input_connections[i];
                    auto& in_desc = dst_data.buffer_input_descriptors[i];
                    const uint32_t num_resources =
                        node_data[src_node].output_buffers[src_output_idx].size();
                    const uint32_t resource_idx =
                        (set_idx + num_resources - in_desc.delay) % num_resources;
                    const auto& resource =
                        node_data[src_node].output_buffers[src_output_idx][resource_idx];
                    dst_data.precomputed_input_buffers[set_idx].push_back(resource.buffer);
                }
                // Precompute outputs
                for (auto& images : dst_data.output_images) {
                    dst_data.precomputed_output_images[set_idx].push_back(
                        images[set_idx % images.size()].image);
                }
                for (auto& buffers : dst_data.output_buffers) {
                    dst_data.precomputed_output_buffers[set_idx].push_back(
                        buffers[set_idx % buffers.size()].buffer);
                }
            }
        }
    }
};

} // namespace merian
