#include "merian/vk/graph/graph.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "vk/utils/math.hpp"

namespace merian {

Graph::Graph(const SharedContext context,
             const ResourceAllocatorHandle allocator,
             const std::optional<QueueHandle> wait_queue)
    : context(context), allocator(allocator), wait_queue(wait_queue),
      debug_utils(context->get_extension<ExtensionVkDebugUtils>()) {}

void Graph::add_node(const std::string& name, const std::shared_ptr<Node>& node) {
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
    node_data[node].image_input_connections.assign(image_inputs.size(), {});
    node_data[node].buffer_input_connections.assign(buffer_inputs.size(), {});
}

void Graph::connect_image(const NodeHandle& src,
                          const NodeHandle& dst,
                          const std::string& src_output,
                          const std::string& dst_input) {
    assert(node_data.contains(src));
    assert(node_data.contains(dst));

    node_data[src].image_connections.insert({dst, src_output, dst_input});
}

void Graph::connect_buffer(const NodeHandle& src,
                           const NodeHandle& dst,
                           const std::string& src_output,
                           const std::string& dst_input) {
    assert(node_data.contains(src));
    assert(node_data.contains(dst));

    node_data[src].buffer_connections.insert({dst, src_output, dst_input});
}

std::vector<NodeHandle> Graph::connect_nodes() {
    // combine connect_*, validate_inputs and calculate_outputs

    std::vector<NodeHandle> topological_order;
    topological_order.reserve(node_data.size());

    topological_visit([&](NodeHandle& node, NodeData& data) {
        topological_order.emplace_back(node);

        // All inputs are connected, i.e. *_input_connections and *_input_descriptors are valid.
        // That means we can compute the nodes' outputs and fill in inputs
        // of the following nodes.

        // 1. Get node output descriptors
        compute_node_output_descriptors(node, data);

        // 2. Resize the output arrays accordingly
        data.image_output_connections.resize(data.image_output_descriptors.size());
        data.buffer_output_connections.resize(data.buffer_output_descriptors.size());

        // 3. Connect outputs to the inputs of dst nodes (fill in their *_input_connections and
        // current *_output_connections).
        for (const NodeConnection& connection : data.image_connections) {
            NodeData& dst_data = node_data[connection.dst];
            const uint32_t src_output_index = data.get_image_output_by_name(connection.src_output);
            const uint32_t dst_input_index = dst_data.get_image_input_by_name(connection.dst_input);

            if (std::get<0>(dst_data.image_input_connections[dst_input_index])) {
                throw std::invalid_argument{
                    fmt::format("The image input '{}' on node '{}' is already connected",
                                connection.dst_input, dst_data.name)};
            }
            if (node == connection.dst &&
                dst_data.image_input_descriptors[dst_input_index].delay == 0) {
                throw std::runtime_error{fmt::format(
                    "node '{}'' is connected to itself {} -> {} with delay 0, maybe you want "
                    "to use a persistent output?",
                    dst_data.name, connection.src_output, connection.dst_input)};
            }
            dst_data.image_input_connections[dst_input_index] = {node, src_output_index};
            data.image_output_connections[src_output_index].emplace_back(connection.dst,
                                                                         dst_input_index);

            // make sure the same underlying resource is not accessed with different layouts
            // from a single node downstream (we can only provide a single layout for cmd_run of the
            // node).
            for (auto& [n, i] : data.image_output_connections[src_output_index]) {
                if (n == connection.dst &&
                    dst_data.image_input_descriptors[i].delay ==
                        dst_data.image_input_descriptors[dst_input_index].delay &&
                    dst_data.image_input_descriptors[i].required_layout !=
                        dst_data.image_input_descriptors[dst_input_index].required_layout) {
                    throw std::invalid_argument{fmt::format(
                        "You are trying to access the same underlying image of node '{}' twice "
                        "from "
                        "node '{}' with connections {} -> {}, {} -> {} and different layouts",
                        data.name, dst_data.name, src_output_index, i, src_output_index,
                        dst_input_index)};
                }
            }
        }
        for (const NodeConnection& connection : data.buffer_connections) {
            NodeData& dst_data = node_data[connection.dst];
            const uint32_t src_output_index = data.get_buffer_output_by_name(connection.src_output);
            const uint32_t dst_input_index =
                dst_data.get_buffer_input_by_name(connection.dst_input);

            if (std::get<0>(dst_data.buffer_input_connections[dst_input_index])) {
                throw std::invalid_argument{
                    fmt::format("The buffer input '{}' on node '{}' is already connected",
                                connection.dst_input, dst_data.name)};
            }
            if (node == connection.dst &&
                dst_data.buffer_input_descriptors[dst_input_index].delay == 0) {
                throw std::runtime_error{fmt::format(
                    "node '{}'' is connected to itself {} -> {} with delay 0, maybe you want "
                    "to use a persistent output?",
                    dst_data.name, connection.src_output, connection.dst_input)};
            }
            dst_data.buffer_input_connections[dst_input_index] = {node, src_output_index};
            data.buffer_output_connections[src_output_index].emplace_back(connection.dst,
                                                                          dst_input_index);
        }
    });

    return topological_order;
}

uint32_t Graph::topological_visit(const std::function<void(NodeHandle&, NodeData&)> visitor) {
    std::unordered_set<NodeHandle> visited;
    std::queue<NodeHandle> queue = start_nodes();

    uint32_t visited_nodes = 0;
    while (!queue.empty()) {
        NodeData& data = node_data[queue.front()];

        visitor(queue.front(), data);
        visited.insert(queue.front());

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
            for (uint32_t i = 0; i < candidate_data.image_input_descriptors.size(); i++) {
                auto& [src_node, src_output_idx] = candidate_data.image_input_connections[i];
                auto& in_desc = candidate_data.image_input_descriptors[i];
                // src was is already processed, or loop with delay > 0.
                satisfied &= visited.contains(src_node) || in_desc.delay > 0;
            }
            for (uint32_t i = 0; i < candidate_data.buffer_input_descriptors.size(); i++) {
                auto& [src_node, src_output_idx] = candidate_data.buffer_input_connections[i];
                auto& in_desc = candidate_data.buffer_input_descriptors[i];
                // src was is already processed, or src == candindate -> self loop with delay > 0.
                satisfied &= visited.contains(src_node) || in_desc.delay > 0;
            }
            if (satisfied) {
                queue.push(candidate);
            }
        }
        queue.pop();
        visited_nodes++;
    }

    return visited_nodes;
}

void Graph::print_error_missing_inputs() {
    for (auto& [dst_node, dst_data] : node_data) {
        // Images
        for (uint32_t i = 0; i < dst_data.image_input_descriptors.size(); i++) {
            auto& [src_node, src_connection_idx] = dst_data.image_input_connections[i];
            auto& in_desc = dst_data.image_input_descriptors[i];
            if (src_node == nullptr) {
                SPDLOG_ERROR(fmt::format("image input '{}' ({}) of node '{}' was not connected!",
                                         in_desc.name, i, dst_data.name));
            }
        }
        // Buffers
        for (uint32_t i = 0; i < dst_data.buffer_input_descriptors.size(); i++) {
            auto& [src_node, src_connection_idx] = dst_data.buffer_input_connections[i];
            auto& in_desc = dst_data.buffer_input_descriptors[i];
            if (src_node == nullptr) {
                SPDLOG_ERROR(fmt::format("buffer input {} ({}) of node {} was not connected!",
                                         in_desc.name, i, dst_data.name));
            }
        }
    }
}

void Graph::cmd_build(vk::CommandBuffer& cmd, const ProfilerHandle profiler) {
    // no nodes -> no build necessary
    if (node_data.empty())
        return;

    // Make sure resources are not in use
    if (wait_queue.has_value()) {
        wait_queue.value()->wait_idle();
    } else {
        context->device.waitIdle();
    }

    reset_graph();

    flat_topology = connect_nodes();
    if (flat_topology.size() != node_data.size()) {
        SPDLOG_ERROR("Graph not fully connected.");
        print_error_missing_inputs();
        throw std::runtime_error{"Graph not fully connected."};
    }

#ifndef NDEBUG
    for (auto& node : flat_topology) {
        SPDLOG_DEBUG("{}", connections(node));
    }
#endif

    allocate_outputs();

    prepare_resource_sets();

    for (auto& node : flat_topology) {
        NodeData& data = node_data[node];
        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, fmt::format("{} ({})", data.name, node->name()));
        SPDLOG_DEBUG("cmd_build node: {} ({})", data.name, node->name());
        cmd_build_node(cmd, node);
    }

    current_iteration = 0;
}

const GraphRun& Graph::cmd_run(vk::CommandBuffer& cmd, const ProfilerHandle profiler) {
    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Graph: run");

    do {
        if (rebuild_requested) {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Graph: build");
            cmd_build(cmd, profiler);
            rebuild_requested = false;
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "Graph: preprocess nodes");
            for (auto& [node, data] : node_data) {
                MERIAN_PROFILE_SCOPE(profiler, fmt::format("{} ({})", data.name, node->name()));
                data.status = {};
                node->pre_process(current_iteration, data.status);
                rebuild_requested |= data.status.request_rebuild;
            }
        }
    } while (rebuild_requested);

    run.reset(current_iteration, profiler, debug_utils);
    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Graph: run nodes");
        for (auto& node : flat_topology) {
            NodeData& data = node_data[node];
            if (data.status.skip_run) {
                continue;
            }
            if (debug_utils)
                debug_utils->cmd_begin_label(cmd, node->name());

            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd,
                                     fmt::format("{} ({})", data.name, node->name()));
            cmd_run_node(cmd, node, data);

            if (debug_utils)
                debug_utils->cmd_end_label(cmd);
        }
    }

    rebuild_requested = run.rebuild_requested;
    current_iteration++;

    return run;
}

std::queue<NodeHandle> Graph::start_nodes() {
    std::queue<NodeHandle> queue;

    // Find nodes without inputs or with delayed inputs only
    for (auto& [node, node_data] : node_data) {
        if (node_data.image_input_descriptors.empty() &&
            node_data.buffer_input_descriptors.empty()) {
            queue.push(node);
            continue;
        }

        if (std::all_of(node_data.image_input_descriptors.begin(),
                        node_data.image_input_descriptors.end(),
                        [](NodeInputDescriptorImage& desc) { return desc.delay > 0; }) &&
            std::all_of(node_data.buffer_input_descriptors.begin(),
                        node_data.buffer_input_descriptors.end(),
                        [](NodeInputDescriptorBuffer& desc) { return desc.delay > 0; })) {
            queue.push(node);
        }
    }
    return queue;
}

void Graph::compute_node_output_descriptors(NodeHandle& node, NodeData& data) {
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
                node_data[src_node].image_output_descriptors[src_output_idx]);
    }
    for (uint32_t i = 0; i < data.buffer_input_descriptors.size(); i++) {
        auto& [src_node, src_output_idx] = data.buffer_input_connections[i];
        auto& in_desc = data.buffer_input_descriptors[i];
        if (in_desc.delay > 0)
            connected_buffer_outputs.push_back(Node::FEEDBACK_OUTPUT_BUFFER);
        else
            connected_buffer_outputs.push_back(
                node_data[src_node].buffer_output_descriptors[src_output_idx]);
    }

    // get outputs from node
    std::tie(data.image_output_descriptors, data.buffer_output_descriptors) =
        node->describe_outputs(connected_image_outputs, connected_buffer_outputs);
}

std::string Graph::connections(NodeHandle& src) {
    std::string result;

    NodeData& src_data = node_data[src];
    for (uint32_t i = 0; i < src_data.image_output_descriptors.size(); i++) {
        auto& src_out_desc = src_data.image_output_descriptors[i];
        auto& src_output = src_data.image_output_connections[i];
        for (auto& [dst_node, image_input_idx] : src_output) {
            NodeData& dst_data = node_data[dst_node];
            auto& dst_in_desc = dst_data.image_input_descriptors[image_input_idx];
            result +=
                fmt::format("image: {} ({}) --{}-> {} ({})\n", src_data.name, src_out_desc.name,
                            dst_in_desc.delay, dst_data.name, dst_in_desc.name);
        }
    }
    for (uint32_t i = 0; i < src_data.buffer_output_descriptors.size(); i++) {
        auto& src_out_desc = src_data.buffer_output_descriptors[i];
        auto& src_output = src_data.buffer_output_connections[i];
        for (auto& [dst_node, buffer_input_idx] : src_output) {
            NodeData& dst_data = node_data[dst_node];
            auto& dst_in_desc = dst_data.buffer_input_descriptors[buffer_input_idx];
            result +=
                fmt::format("buffer: {} ({}) --{}-> {} ({})\n", src_data.name, src_out_desc.name,
                            dst_in_desc.delay, dst_data.name, dst_in_desc.name);
        }
    }

    return result;
}

void Graph::allocate_outputs() {
    for (auto& [src_node, src_data] : node_data) {
        // Buffers
        src_data.allocated_buffer_outputs.resize(src_data.buffer_output_descriptors.size());
        for (uint32_t src_out_idx = 0; src_out_idx < src_data.buffer_output_descriptors.size();
             src_out_idx++) {
            auto& out_desc = src_data.buffer_output_descriptors[src_out_idx];
            vk::BufferUsageFlags usage_flags = out_desc.create_info.usage;
            vk::PipelineStageFlags2 input_pipeline_stages;
            vk::AccessFlags2 input_access_flags;
            uint32_t max_delay = 0;
            for (auto& [dst_node, dst_input_idx] :
                 src_data.buffer_output_connections[src_out_idx]) {
                auto& in_desc = node_data[dst_node].buffer_input_descriptors[dst_input_idx];
                if (out_desc.persistent && in_desc.delay > 0) {
                    throw std::runtime_error{fmt::format(
                        "persistent outputs cannot be accessed with delay > 0. {}: {} -> {}: "
                        "{}",
                        src_data.name, src_out_idx, node_data[dst_node].name, dst_input_idx)};
                }
                max_delay = std::max(max_delay, in_desc.delay);
                usage_flags |= in_desc.usage_flags;
                input_pipeline_stages |= in_desc.pipeline_stages;
                input_access_flags |= in_desc.access_flags;
            }
            // Create max_delay + 1 buffers
            for (uint32_t j = 0; j < max_delay + 1; j++) {
                BufferHandle buffer =
                    allocator->createBuffer(out_desc.create_info.size, usage_flags, NONE,
                                            fmt::format("node '{}' buffer, output '{}', copy '{}'",
                                                        src_data.name, out_desc.name, j));
                src_data.allocated_buffer_outputs[src_out_idx].emplace_back(
                    std::make_shared<BufferResource>(buffer, vk::PipelineStageFlagBits2::eTopOfPipe,
                                                     vk::AccessFlags2(), false,
                                                     input_pipeline_stages, input_access_flags));
            }
        }

        // Images
        src_data.allocated_image_outputs.resize(src_data.image_output_descriptors.size());
        for (uint32_t src_out_idx = 0; src_out_idx < src_data.image_output_descriptors.size();
             src_out_idx++) {
            auto& out_desc = src_data.image_output_descriptors[src_out_idx];
            vk::ImageCreateInfo create_info = out_desc.create_info;
            vk::PipelineStageFlags2 input_pipeline_stages;
            vk::AccessFlags2 input_access_flags;
            uint32_t max_delay = 0;
            for (auto& [dst_node, dst_input_idx] : src_data.image_output_connections[src_out_idx]) {
                auto& in_desc = node_data[dst_node].image_input_descriptors[dst_input_idx];
                if (out_desc.persistent && in_desc.delay > 0) {
                    throw std::runtime_error{fmt::format(
                        "persistent outputs cannot be accessed with delay > 0. {}: {} -> {}: "
                        "{}",
                        src_data.name, src_out_idx, node_data[dst_node].name, dst_input_idx)};
                }
                max_delay = std::max(max_delay, in_desc.delay);
                create_info.usage |= in_desc.usage_flags;
                input_pipeline_stages |= in_desc.pipeline_stages;
                input_access_flags |= in_desc.access_flags;
            }
            // Create max_delay + 1 images
            for (uint32_t j = 0; j < max_delay + 1; j++) {
                ImageHandle image =
                    allocator->createImage(create_info, NONE,
                                           fmt::format("node '{}' image, output '{}', copy '{}'",
                                                       src_data.name, out_desc.name, j));
                src_data.allocated_image_outputs[src_out_idx].emplace_back(
                    std::make_shared<ImageResource>(image, vk::PipelineStageFlagBits2::eTopOfPipe,
                                                    vk::AccessFlags2(), false,
                                                    input_pipeline_stages, input_access_flags));
            }
        }
    }
}

void Graph::prepare_resource_sets() {
    for (auto& [dst_node, dst_data] : node_data) {
        // Find the lowest number of sets needed (lcm)
        std::vector<uint32_t> num_resources;

        // By checking how many copies of that resource exists in the sources
        for (auto& [src_node, src_output_idx] : dst_data.image_input_connections) {
            num_resources.push_back(
                node_data[src_node].allocated_image_outputs[src_output_idx].size());
        }
        for (auto& [src_node, src_output_idx] : dst_data.buffer_input_connections) {
            num_resources.push_back(
                node_data[src_node].allocated_buffer_outputs[src_output_idx].size());
        }
        // ...and how many output resources the node has
        for (auto& images : dst_data.allocated_image_outputs) {
            num_resources.push_back(images.size());
        }
        for (auto& buffers : dst_data.allocated_buffer_outputs) {
            num_resources.push_back(buffers.size());
        }

        // After this many iterations we can again use the first resource set
        uint32_t num_sets = lcm(num_resources);

        // Precompute resource sets for each iteration
        dst_data.precomputed_input_images.resize(num_sets);
        dst_data.precomputed_input_buffers.resize(num_sets);
        dst_data.precomputed_output_images.resize(num_sets);
        dst_data.precomputed_output_buffers.resize(num_sets);
        dst_data.precomputed_input_images_resource.resize(num_sets);
        dst_data.precomputed_input_buffers_resource.resize(num_sets);
        dst_data.precomputed_output_images_resource.resize(num_sets);
        dst_data.precomputed_output_buffers_resource.resize(num_sets);

        for (uint32_t set_idx = 0; set_idx < num_sets; set_idx++) {
            // Precompute inputs
            for (uint32_t i = 0; i < dst_data.image_input_descriptors.size(); i++) {
                auto& [src_node, src_output_idx] = dst_data.image_input_connections[i];
                auto& in_desc = dst_data.image_input_descriptors[i];
                const uint32_t num_resources =
                    node_data[src_node].allocated_image_outputs[src_output_idx].size();
                const uint32_t resource_idx =
                    (set_idx + num_resources - in_desc.delay) % num_resources;
                const auto& resource =
                    node_data[src_node].allocated_image_outputs[src_output_idx][resource_idx];
                dst_data.precomputed_input_images[set_idx].push_back(resource->image);
                dst_data.precomputed_input_images_resource[set_idx].push_back(resource);
            }
            for (uint32_t i = 0; i < dst_data.buffer_input_descriptors.size(); i++) {
                auto& [src_node, src_output_idx] = dst_data.buffer_input_connections[i];
                auto& in_desc = dst_data.buffer_input_descriptors[i];
                const uint32_t num_resources =
                    node_data[src_node].allocated_buffer_outputs[src_output_idx].size();
                const uint32_t resource_idx =
                    (set_idx + num_resources - in_desc.delay) % num_resources;
                const auto& resource =
                    node_data[src_node].allocated_buffer_outputs[src_output_idx][resource_idx];
                dst_data.precomputed_input_buffers[set_idx].push_back(resource->buffer);
                dst_data.precomputed_input_buffers_resource[set_idx].push_back(resource);
            }
            // Precompute outputs
            for (auto& images : dst_data.allocated_image_outputs) {
                dst_data.precomputed_output_images[set_idx].push_back(
                    images[set_idx % images.size()]->image);
                dst_data.precomputed_output_images_resource[set_idx].push_back(
                    images[set_idx % images.size()]);
            }
            for (auto& buffers : dst_data.allocated_buffer_outputs) {
                dst_data.precomputed_output_buffers[set_idx].push_back(
                    buffers[set_idx % buffers.size()]->buffer);
                dst_data.precomputed_output_buffers_resource[set_idx].push_back(
                    buffers[set_idx % buffers.size()]);
            }
        }
    }
}

void Graph::cmd_build_node(vk::CommandBuffer& cmd, NodeHandle& node) {
    NodeData& data = node_data[node];
    for (uint32_t set_idx = 0; set_idx < data.precomputed_input_images.size(); set_idx++) {
        cmd_barrier_for_node(cmd, data, set_idx);
    }
    node->cmd_build(cmd, data.precomputed_input_images, data.precomputed_input_buffers,
                    data.precomputed_output_images, data.precomputed_output_buffers);
}

// Insert the according barriers for that node
void Graph::cmd_run_node(vk::CommandBuffer& cmd, NodeHandle& node, NodeData& data) {

    if (data.precomputed_input_images.size()) {
        uint32_t set_idx = current_iteration % data.precomputed_input_images.size();

        cmd_barrier_for_node(cmd, data, set_idx);

        auto& in_images = data.precomputed_input_images[set_idx];
        auto& in_buffers = data.precomputed_input_buffers[set_idx];
        auto& out_images = data.precomputed_output_images[set_idx];
        auto& out_buffers = data.precomputed_output_buffers[set_idx];

        node->cmd_process(cmd, run, set_idx, in_images, in_buffers, out_images, out_buffers);
    } else {
        node->cmd_process(cmd, run, -1, {}, {}, {}, {});
    }
}

void Graph::cmd_barrier_for_node(vk::CommandBuffer& cmd, NodeData& data, uint32_t& set_idx) {
    image_barriers_for_set.clear();
    buffer_barriers_for_set.clear();

    auto& in_images_res = data.precomputed_input_images_resource[set_idx];
    auto& in_buffers_res = data.precomputed_input_buffers_resource[set_idx];

    // in-images
    for (uint32_t i = 0; i < data.image_input_descriptors.size(); i++) {
        auto& in_desc = data.image_input_descriptors[i];
        auto& res = in_images_res[i];
        if (res->last_used_as_output) {
            // Need to insert barrier and transition layout
            vk::ImageMemoryBarrier2 img_bar = res->image->barrier2(
                in_desc.required_layout, res->current_access_flags, res->input_access_flags,
                res->current_stage_flags, res->input_stage_flags);
            image_barriers_for_set.push_back(img_bar);
            res->current_stage_flags = res->input_stage_flags;
            res->current_access_flags = res->input_access_flags;
            res->last_used_as_output = false;
        } else {
            // No barrier required, if no transition required
            if (in_desc.required_layout != res->image->get_current_layout()) {
                vk::ImageMemoryBarrier2 img_bar = res->image->barrier2(
                    in_desc.required_layout, res->current_access_flags, res->current_access_flags,
                    res->current_stage_flags, res->current_stage_flags);
                image_barriers_for_set.push_back(img_bar);
            }
        }
    }
    // in-buffers
    for (uint32_t i = 0; i < data.buffer_input_descriptors.size(); i++) {
        auto& res = in_buffers_res[i];
        if (res->last_used_as_output) {
            vk::BufferMemoryBarrier2 buffer_bar{res->current_stage_flags,
                                                res->current_access_flags,
                                                res->input_stage_flags,
                                                res->input_access_flags,
                                                VK_QUEUE_FAMILY_IGNORED,
                                                VK_QUEUE_FAMILY_IGNORED,
                                                *res->buffer,
                                                0,
                                                VK_WHOLE_SIZE};
            buffer_barriers_for_set.push_back(buffer_bar);
            res->current_stage_flags = res->input_stage_flags;
            res->current_access_flags = res->input_access_flags;
            res->last_used_as_output = false;
        } // else nothing to do
    }

    auto& out_images_res = data.precomputed_output_images_resource[set_idx];
    auto& out_buffers_res = data.precomputed_output_buffers_resource[set_idx];

    // out-images
    for (uint32_t i = 0; i < data.image_output_descriptors.size(); i++) {
        auto& out_desc = data.image_output_descriptors[i];
        auto& res = out_images_res[i];
        // if not persistent: transition from undefined -> a bit faster
        vk::ImageMemoryBarrier2 img_bar = res->image->barrier2(
            out_desc.required_layout, res->current_access_flags, out_desc.access_flags,
            res->current_stage_flags, out_desc.pipeline_stages, VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), !out_desc.persistent);

        image_barriers_for_set.push_back(img_bar);
        res->current_stage_flags = out_desc.pipeline_stages;
        res->current_access_flags = out_desc.access_flags;
        res->last_used_as_output = true;
    }
    // out-buffers
    for (uint32_t i = 0; i < data.buffer_output_descriptors.size(); i++) {
        auto& out_desc = data.buffer_output_descriptors[i];
        auto& res = out_buffers_res[i];

        vk::BufferMemoryBarrier2 buffer_bar{res->current_stage_flags,
                                            res->current_access_flags,
                                            out_desc.pipeline_stages,
                                            out_desc.access_flags,
                                            VK_QUEUE_FAMILY_IGNORED,
                                            VK_QUEUE_FAMILY_IGNORED,
                                            *res->buffer,
                                            0,
                                            VK_WHOLE_SIZE};
        buffer_barriers_for_set.push_back(buffer_bar);
        res->current_stage_flags = out_desc.pipeline_stages;
        res->current_access_flags = out_desc.access_flags;
        res->last_used_as_output = true;
    }

    vk::DependencyInfoKHR dep_info{{}, {}, buffer_barriers_for_set, image_barriers_for_set};
    cmd.pipelineBarrier2(dep_info);
}

void Graph::reset_graph() {
    this->flat_topology.clear();
    for (auto& [node, data] : node_data) {

        data.image_input_connections.assign(data.image_input_descriptors.size(), {});
        data.buffer_input_connections.assign(data.buffer_input_descriptors.size(), {});

        data.image_output_descriptors.clear();
        data.buffer_output_descriptors.clear();
        data.image_output_connections.clear();
        data.buffer_output_connections.clear();

        data.allocated_image_outputs.clear();
        data.allocated_buffer_outputs.clear();

        data.precomputed_input_images.clear();
        data.precomputed_input_buffers.clear();
        data.precomputed_output_images.clear();
        data.precomputed_output_buffers.clear();

        data.precomputed_input_images_resource.clear();
        data.precomputed_input_buffers_resource.clear();
        data.precomputed_output_images_resource.clear();
        data.precomputed_output_buffers_resource.clear();
    }
}

void Graph::get_configuration_io_for_node(Configuration& config, NodeData& data) {
    if ((!data.image_output_descriptors.empty() || !data.buffer_output_descriptors.empty()) &&
        config.st_begin_child("graph_outputs", "Outputs")) {
        for (uint32_t i = 0; i < data.image_output_descriptors.size(); i++) {
            auto& out_desc = data.image_output_descriptors[i];
            if (config.st_begin_child("image_output-" + out_desc.name,
                                      fmt::format("{} (image)", out_desc.name))) {
                for (auto& out : data.image_output_connections[i]) {
                    NodeData& input_data = node_data[std::get<0>(out)];
                    const std::string id =
                        fmt::format("image_input-{}-{}", input_data.name,
                                    input_data.image_input_descriptors[std::get<1>(out)].name);
                    const std::string label = fmt::format(
                        "{}, of {} ({})", input_data.image_input_descriptors[std::get<1>(out)].name,
                        input_data.name, input_data.node->name());
                    if (config.st_begin_child(id, label)) {
                        config.st_end_child();
                    }
                }
                config.st_end_child();
            }
        }
        for (uint32_t i = 0; i < data.buffer_output_descriptors.size(); i++) {
            auto& out_desc = data.buffer_output_descriptors[i];
            if (config.st_begin_child("buffer_output-" + out_desc.name,
                                      fmt::format("{} (buffer)", out_desc.name))) {
                for (auto& out : data.buffer_output_connections[i]) {
                    NodeData& input_data = node_data[std::get<0>(out)];
                    const std::string id =
                        fmt::format("buffer_input-{}-{}", input_data.name,
                                    input_data.buffer_input_descriptors[std::get<1>(out)].name);
                    const std::string label =
                        fmt::format("{}, of {} ({})",
                                    input_data.buffer_input_descriptors[std::get<1>(out)].name,
                                    input_data.name, input_data.node->name());
                    if (config.st_begin_child(id, label)) {
                        config.st_end_child();
                    }
                }
                config.st_end_child();
            }
        }
        config.st_end_child();
    }
    if ((!data.image_input_descriptors.empty() || !data.buffer_input_descriptors.empty()) &&
        config.st_begin_child("graph_inputs", "Inputs")) {
        for (uint32_t i = 0; i < data.image_input_descriptors.size(); i++) {
            auto& in_desc = data.image_input_descriptors[i];
            uint32_t src_out_index = std::get<1>(data.image_input_connections[i]);
            auto& out_node_data = node_data[std::get<0>(data.image_input_connections[i])];
            if (config.st_begin_child(
                    "image_input-" + in_desc.name,
                    fmt::format("{} <-image- {}, from {} ({})", in_desc.name,
                                out_node_data.image_output_descriptors[src_out_index].name,
                                out_node_data.name, out_node_data.node->name()))) {
                config.st_end_child();
            }
        }
        for (uint32_t i = 0; i < data.buffer_input_descriptors.size(); i++) {
            auto& in_desc = data.buffer_input_descriptors[i];
            uint32_t src_out_index = std::get<1>(data.buffer_input_connections[i]);
            auto& out_node_data = node_data[std::get<0>(data.buffer_input_connections[i])];
            if (config.st_begin_child(
                    "buffer_input-" + in_desc.name,
                    fmt::format("{} <-buffer- {}, from {} ({})", in_desc.name,
                                out_node_data.buffer_output_descriptors[src_out_index].name,
                                out_node_data.name, out_node_data.node->name()))) {
                config.st_end_child();
            }
        }
        config.st_end_child();
    }
}

void Graph::get_configuration(Configuration& config) {
    if (config.st_new_section("Graph")) {
        rebuild_requested |= config.config_bool("Rebuild");
        config.st_no_space();
        config.output_text(fmt::format("Current iteration {}", current_iteration));

        for (auto& [node, data] : node_data) {
            std::string node_label = fmt::format("{} ({})", data.name, data.node->name());
            if (config.st_begin_child(data.name.c_str(), node_label.c_str())) {
                bool needs_rebuild = false;
                data.node->get_configuration(config, needs_rebuild);
                rebuild_requested |= needs_rebuild;
                config.st_separate();
                get_configuration_io_for_node(config, data);
                config.st_end_child();
            }
        }
    }
}

} // namespace merian
