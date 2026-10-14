#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_in.hpp"
#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/connectors/image/vk_image_in.hpp"
#include "merian-graph/connectors/image/vk_image_out_external.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/ptr_out.hpp"
#include "merian-graph/graph/graph.hpp"
#include "merian/utils/properties_json_load.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>

using namespace merian;
using json = nlohmann::json;

inline uint32_t
pattern_value(const uint32_t id, const std::string& output, const uint64_t iteration) {
    const std::array<std::string, 4> outputs = {"out", "buf", "inp", "binp"};
    const auto tag = static_cast<uint32_t>(std::ranges::find(outputs, output) - outputs.begin());
    return (id << 20) | (tag << 18) | static_cast<uint32_t>(iteration & 0x3FFFF);
}

// What an output holds before its node first wrote it.
inline uint32_t initial_pattern_value(const uint32_t id, const std::string& output) {
    return pattern_value(id, output, 0x3FFFF);
}

constexpr uint32_t IMAGE_INPUT_COUNT = 3;
constexpr uint32_t BUFFER_INPUT_COUNT = 2;
constexpr uint32_t INPUT_COUNT = IMAGE_INPUT_COUNT + BUFFER_INPUT_COUNT;
inline const std::array<std::string, INPUT_COUNT> INPUT_NAMES = {"in0", "in1", "in2", "bin0",
                                                                 "bin1"};

inline bool is_image_input(const std::string& input) {
    return !input.starts_with("b");
}

struct PatternShape {
    uint32_t width = 16;
    uint32_t height = 16;
    uint32_t buffer_words = 1024;
    bool persistent = false;
    bool scratch = false;
    std::array<uint32_t, INPUT_COUNT> delays{};
    std::array<bool, INPUT_COUNT> required{};
    bool pass_through = false;
    bool in_place = false;
    bool required_in_place = false;
    bool fail_outputs = false;
    bool fail_connected = false;
    bool fail_process = false;
    bool counting = false;
    bool sampled = false;
    bool reconnect_on_change = false;
    bool defaults = false;

    bool operator==(const PatternShape&) const = default;
};

// Fills every copy with a value on connect.
class InitializedImageOut : public ManagedVkImageOut {
  public:
    InitializedImageOut(const vk::ImageCreateInfo& create_info, const uint32_t value)
        : ManagedVkImageOut(create_info), value(value) {}

    void on_connected(const ResourceConnectInfo& info,
                      Submission& submission,
                      const GraphResourceHandle& resource) override {
        ManagedVkImageOut::on_connected(info, submission, resource);
        const CommandBufferHandle& cmd = submission.get_cmd();
        const ImageHandle& image = this->resource(resource).get_image();
        cmd->barrier(image->barrier2(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eMemoryWrite,
                                     vk::AccessFlagBits2::eTransferWrite,
                                     vk::PipelineStageFlagBits2::eAllCommands,
                                     vk::PipelineStageFlagBits2::eTransfer, VK_QUEUE_FAMILY_IGNORED,
                                     VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), true));
        cmd->clear(image, vk::ClearColorValue{std::array<uint32_t, 4>{value, value, value, value}});
        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite});
    }

  private:
    const uint32_t value;
};

// Fills every copy with a value on connect.
class InitializedBufferOut : public ManagedVkBufferOut {
  public:
    InitializedBufferOut(const vk::BufferCreateInfo& create_info, const uint32_t value)
        : ManagedVkBufferOut(create_info), value(value) {}

    void on_connected(const ResourceConnectInfo& info,
                      Submission& submission,
                      const GraphResourceHandle& resource) override {
        ManagedVkBufferOut::on_connected(info, submission, resource);
        const CommandBufferHandle& cmd = submission.get_cmd();
        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite});
        cmd->fill(this->resource(resource).get_buffer(0), value);
        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite});
    }

  private:
    const uint32_t value;
};

class CountingImageOut : public InitializedImageOut {
  public:
    using InitializedImageOut::InitializedImageOut;

    GraphResourceHandle create_resource(const ResourceCreateInfo& info) override {
        created++;
        access = info.combined_access;
        return InitializedImageOut::create_resource(info);
    }

    void on_connected(const ResourceConnectInfo& info,
                      Submission& submission,
                      const GraphResourceHandle& resource) override {
        connected++;
        InitializedImageOut::on_connected(info, submission, resource);
    }

    ConnectorStatusFlags
    on_pre_process(Submission& submission,
                   const GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override {
        pre_processed++;
        return ManagedVkImageOut::on_pre_process(submission, resource, node, image_barriers,
                                                 buffer_barriers);
    }

    ConnectorStatusFlags
    on_post_process(Submission& submission,
                    const GraphResourceHandle& resource,
                    const NodeHandle& node,
                    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override {
        post_processed++;
        return ManagedVkImageOut::on_post_process(submission, resource, node, image_barriers,
                                                  buffer_barriers);
    }

    uint32_t created = 0;
    uint32_t connected = 0;
    uint32_t pre_processed = 0;
    uint32_t post_processed = 0;
    ConnectorAccess access;
};

class PatternNode : public Node {
  public:
    static constexpr uint32_t MAX_EXTENT = 64;
    static constexpr vk::DeviceSize SLOT_SIZE =
        vk::DeviceSize{MAX_EXTENT} * MAX_EXTENT * sizeof(uint32_t);
    static constexpr uint32_t SLOT_COUNT = 160;

    struct Record {
        std::string input;
        uint64_t total_iteration;
        uint64_t iteration;
        uint32_t slot;
        uint32_t word_count;
    };

    void initialize(const ContextHandle& /*context*/,
                    const ResourceAllocatorHandle& allocator) override {
        initialize_count++;
        readback =
            allocator->create_buffer(SLOT_SIZE * SLOT_COUNT, vk::BufferUsageFlagBits::eTransferDst,
                                     MemoryMappingType::HOST_ACCESS_RANDOM, "readback");
    }

    std::vector<InputConnectorDescriptor> describe_inputs() override {
        std::vector<InputConnectorDescriptor> inputs;
        for (uint32_t i = 0; i < IMAGE_INPUT_COUNT; i++) {
            inputs.push_back({.name = INPUT_NAMES[i],
                              .connector = con_images[i],
                              .access = shape.sampled && i == 0 ? ConnectorAccess::transfer_src |
                                                                      ConnectorAccess::compute_read
                                                                : ConnectorAccess::transfer_src,
                              .delay = shape.delays[i],
                              .optional = !shape.required[i],
                              .pass_through = shape.pass_through && i == 0 ? "out" : "",
                              .auto_connect = shape.defaults && i == 1 && shape.delays[i] > 0
                                                  ? AutoConnect::output("out")
                                                  : AutoConnect{}});
        }
        for (uint32_t i = 0; i < BUFFER_INPUT_COUNT; i++) {
            const uint32_t input = IMAGE_INPUT_COUNT + i;
            inputs.push_back({.name = INPUT_NAMES[input],
                              .connector = con_buffers[i],
                              .access = ConnectorAccess::transfer_src,
                              .delay = shape.delays[input],
                              .optional = !shape.required[input],
                              .pass_through = shape.pass_through && i == 0 ? "buf" : "",
                              .auto_connect = shape.defaults && i == 1 && shape.delays[input] > 0
                                                  ? AutoConnect::input("bin0")
                                                  : AutoConnect{}});
        }
        if (shape.in_place) {
            inputs.push_back(
                {.name = "inp",
                 .connector = con_in_place,
                 .access = ConnectorAccess::transfer_src | ConnectorAccess::transfer_dst,
                 .optional = !shape.required_in_place,
                 .in_place = true});
            inputs.push_back(
                {.name = "binp",
                 .connector = con_buffer_in_place,
                 .access = ConnectorAccess::transfer_src | ConnectorAccess::transfer_dst,
                 .optional = !shape.required_in_place,
                 .in_place = true});
        }
        return inputs;
    }

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        if (shape.fail_outputs) {
            throw graph_errors::node_error{"failing on purpose"};
        }
        const vk::ImageCreateInfo out_create_info =
            *ManagedVkImageOut::create(vk::Format::eR32Uint, shape.width, shape.height)
                 ->get_create_info();
        if (shape.counting) {
            counting = std::make_shared<CountingImageOut>(out_create_info,
                                                          initial_pattern_value(id, "out"));
            con_out = counting;
        } else {
            con_out = std::make_shared<InitializedImageOut>(out_create_info,
                                                            initial_pattern_value(id, "out"));
        }
        con_buf = std::make_shared<InitializedBufferOut>(
            vk::BufferCreateInfo{{},
                                 shape.buffer_words * sizeof(uint32_t),
                                 vk::BufferUsageFlagBits::eTransferSrc |
                                     vk::BufferUsageFlagBits::eTransferDst},
            initial_pattern_value(id, "buf"));
        std::vector<OutputConnectorDescriptor> outputs = {
            {.name = "out",
             .connector = con_out,
             .access = ConnectorAccess::transfer_dst,
             .persistent = shape.persistent},
            {.name = "buf",
             .connector = con_buf,
             .access = ConnectorAccess::transfer_dst,
             .persistent = shape.persistent},
        };
        if (shape.scratch) {
            con_tmp = ManagedVkImageOut::create(vk::Format::eR32Uint, shape.width, shape.height);
            outputs.push_back(
                {.name = "tmp",
                 .connector = con_tmp,
                 .access = ConnectorAccess::transfer_src | ConnectorAccess::transfer_dst,
                 .scratch = true});
        }
        return outputs;
    }

    NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                 const NodeIO& /*io*/,
                                 const NodeConnectionInfo& /*info*/,
                                 Submission& /*submission*/) override {
        if (shape.fail_connected) {
            throw graph_errors::node_error{"failing on purpose"};
        }
        if (shape.reconnect_on_change && io_layout.is_connected(con_out) != out_connected) {
            out_connected = !out_connected;
            return NEEDS_RECONNECT;
        }
        return {};
    }

    NodeStatusFlags pre_process(const NodeIO& /*io*/, const NodeProcessInfo& /*info*/) override {
        if (pending_shape) {
            shape = *pending_shape;
            pending_shape.reset();
            return NEEDS_RECONNECT;
        }
        return {};
    }

    NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override {
        if (shape.fail_process) {
            throw graph_errors::node_error{"failing on purpose"};
        }
        process_count++;
        const CommandBufferHandle& cmd = submission.get_cmd();
        const auto value = [&](const std::string& output) {
            return pattern_value(id, output, info.get_total_iteration());
        };
        const auto clear_value = [&](const std::string& output) {
            const uint32_t word = value(output);
            return vk::ClearColorValue{std::array<uint32_t, 4>{word, word, word, word}};
        };

        // first: an input wrongly sharing memory with the scratch image reads this node's value
        if (shape.scratch) {
            cmd->clear(io[con_tmp], clear_value("out"));
        }

        const auto add_record = [&](const std::string& input, const uint32_t word_count) {
            if (records.size() == SLOT_COUNT) {
                throw std::runtime_error{"readback slots exhausted"};
            }
            const uint32_t slot = static_cast<uint32_t>(records.size());
            records.push_back(
                {input, info.get_total_iteration(), info.get_iteration(), slot, word_count});
            return slot * SLOT_SIZE;
        };
        for (uint32_t i = 0; i < IMAGE_INPUT_COUNT; i++) {
            if (!io.is_connected(con_images[i])) {
                continue;
            }
            const ImageHandle& image = io[con_images[i]].get_image();
            if (!image) {
                continue;
            }
            const vk::Extent3D extent = image->get_extent();
            const vk::DeviceSize offset = add_record(INPUT_NAMES[i], extent.width * extent.height);
            cmd->copy(image, readback,
                      vk::BufferImageCopy{
                          offset, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {}, extent});
        }
        for (uint32_t i = 0; i < BUFFER_INPUT_COUNT; i++) {
            if (!io.is_connected(con_buffers[i])) {
                continue;
            }
            const BufferHandle& buffer = io[con_buffers[i]];
            const vk::DeviceSize size = std::min(buffer->get_size(), SLOT_SIZE);
            const vk::DeviceSize offset = add_record(
                INPUT_NAMES[IMAGE_INPUT_COUNT + i], static_cast<uint32_t>(size / sizeof(uint32_t)));
            cmd->copy(buffer, readback, vk::BufferCopy{0, offset, size});
        }
        if (shape.in_place && io.is_connected(con_in_place)) {
            const ImageHandle& image = io[con_in_place];
            const vk::Extent3D extent = image->get_extent();
            const vk::DeviceSize offset = add_record("inp", extent.width * extent.height);
            cmd->copy(image, readback,
                      vk::BufferImageCopy{
                          offset, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {}, extent});
            cmd->barrier(image->barrier2(
                vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eTransferRead,
                vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
                vk::PipelineStageFlagBits2::eTransfer));
            cmd->clear(image, clear_value("inp"));
        }
        if (shape.in_place && io.is_connected(con_buffer_in_place)) {
            const BufferHandle& buffer = io[con_buffer_in_place];
            const vk::DeviceSize size = std::min(buffer->get_size(), SLOT_SIZE);
            const vk::DeviceSize offset =
                add_record("binp", static_cast<uint32_t>(size / sizeof(uint32_t)));
            cmd->copy(buffer, readback, vk::BufferCopy{0, offset, size});
            cmd->barrier(vk::MemoryBarrier2{
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite});
            cmd->fill(buffer, value("binp"));
        }

        if (shape.scratch) {
            const ImageHandle& tmp = io[con_tmp];
            cmd->barrier(tmp->barrier2(
                vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eTransferWrite,
                vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer,
                vk::PipelineStageFlagBits2::eTransfer));
            cmd->copy(tmp, io[con_out]);
        } else {
            cmd->clear(io[con_out], clear_value("out"));
        }
        cmd->fill(io[con_buf], value("buf"));
        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead});
        return {};
    }

    NodeStatusFlags properties(Properties& props) override {
        PatternShape new_shape = shape;
        props.config_uint("id", id);
        props.config_uint("width", new_shape.width);
        props.config_uint("height", new_shape.height);
        props.config_uint("buffer words", new_shape.buffer_words);
        props.config_bool("persistent", new_shape.persistent);
        props.config_bool("scratch", new_shape.scratch);
        props.config_bool("pass through", new_shape.pass_through);
        props.config_bool("in place", new_shape.in_place);
        props.config_bool("required in place", new_shape.required_in_place);
        props.config_bool("fail outputs", new_shape.fail_outputs);
        props.config_bool("fail connected", new_shape.fail_connected);
        props.config_bool("fail process", new_shape.fail_process);
        props.config_bool("counting", new_shape.counting);
        props.config_bool("sampled", new_shape.sampled);
        props.config_bool("reconnect on change", new_shape.reconnect_on_change);
        props.config_bool("defaults", new_shape.defaults);
        for (uint32_t i = 0; i < INPUT_COUNT; i++) {
            props.config_uint("delay " + INPUT_NAMES[i], new_shape.delays[i]);
            props.config_bool("required " + INPUT_NAMES[i], new_shape.required[i]);
        }
        if (new_shape != shape) {
            shape = new_shape;
            return NEEDS_RECONNECT;
        }
        return {};
    }

    void set_shape(const PatternShape& new_shape) {
        pending_shape = new_shape;
    }

    std::vector<uint32_t> read(const Record& record) const {
        const MemoryAllocationHandle& memory = readback->get_memory();
        memory->invalidate(record.slot * SLOT_SIZE, record.word_count * sizeof(uint32_t));
        const uint32_t* data =
            memory->map_as<uint32_t>() + record.slot * (SLOT_SIZE / sizeof(uint32_t));
        std::vector<uint32_t> words(data, data + record.word_count);
        memory->unmap();
        return words;
    }

    uint32_t id = 0;
    PatternShape shape;
    std::vector<Record> records;
    uint32_t process_count = 0;
    uint32_t initialize_count = 0;
    std::shared_ptr<CountingImageOut> counting;

  private:
    std::optional<PatternShape> pending_shape;
    bool out_connected = false;

    std::array<VkImageInHandle, IMAGE_INPUT_COUNT> con_images = {
        VkImageIn::create(), VkImageIn::create(), VkImageIn::create()};
    std::array<VkBufferInHandle, BUFFER_INPUT_COUNT> con_buffers = {VkBufferIn::create(),
                                                                    VkBufferIn::create()};
    VkImageInHandle con_in_place = VkImageIn::create();
    VkBufferInHandle con_buffer_in_place = VkBufferIn::create();
    ManagedVkImageOutHandle con_out;
    ManagedVkBufferOutHandle con_buf;
    ManagedVkImageOutHandle con_tmp;

    BufferHandle readback;
};

class CanvasSource : public Node {
  public:
    void initialize(const ContextHandle& /*context*/,
                    const ResourceAllocatorHandle& allocator) override {
        const ImageHandle image =
            allocator->create_image(vk::ImageCreateInfo{{},
                                                        vk::ImageType::e2D,
                                                        vk::Format::eR32Uint,
                                                        {16, 16, 1},
                                                        1,
                                                        1,
                                                        vk::SampleCountFlagBits::e1,
                                                        vk::ImageTiling::eOptimal,
                                                        vk::ImageUsageFlagBits::eTransferSrc |
                                                            vk::ImageUsageFlagBits::eTransferDst |
                                                            vk::ImageUsageFlagBits::eSampled},
                                    MemoryMappingType::NONE, "canvas");
        view = ImageView::create(image);
    }

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        return {{"canvas", con_canvas, ConnectorAccess::transfer_dst}};
    }

    NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& /*submission*/) override {
        io[con_canvas].set(gaps && info.get_iteration() % 2 == 1 ? nullptr : view);
        return {};
    }

    bool gaps = false;

  private:
    ImageViewHandle view;
    const ExternalVkImageOutHandle con_canvas = ExternalVkImageOut::create();
};

class PatternGroup : public Node {
  public:
    enum class Broken : uint32_t {
        NO,
        MISSING_INPUT,
        UNKNOWN_NODE,
        THROWS,
    };

    struct Settings {
        std::string prefix = "c";
        uint32_t broken = static_cast<uint32_t>(Broken::NO);
        bool parallel = false;
        bool public_in1 = false;
        std::string own_input;
        uint32_t own_delay = 0;
        bool own_required = false;
        bool own_output = false;

        bool operator==(const Settings&) const = default;
    };

    Subgraph describe_subgraph() override {
        if (settings.broken == static_cast<uint32_t>(Broken::THROWS)) {
            throw graph_errors::node_error{"failing on purpose"};
        }
        Subgraph subgraph;
        std::vector<std::string> stages;
        for (uint32_t i = 0; i < children.size(); i++) {
            stages.push_back(fmt::format("{}{}", settings.prefix, i));
            subgraph.add_node(stages.back(), children[i]);
        }
        if (inner) {
            stages.push_back(fmt::format("{}{}", settings.prefix, children.size()));
            subgraph.add_node(stages.back(), inner);
        }
        std::string previous;
        for (const std::string& stage : stages) {
            const bool first = previous.empty() || settings.parallel;
            subgraph.add_connection(first ? ".in0" : previous + ".out", stage + ".in0");
            subgraph.add_connection(first ? ".bin0" : previous + ".buf", stage + ".bin0");
            subgraph.add_connection(".in2", stage + ".in2");
            previous = stage;
        }
        if (!previous.empty()) {
            subgraph.add_connection(previous + ".out", ".out");
            subgraph.add_connection(previous + ".buf", ".buf");
            subgraph.add_pass_through(".in0", ".out");
            subgraph.add_pass_through(".bin0", ".buf");
        }
        if (settings.public_in1 && !children.empty()) {
            subgraph.add_connection(".in2", stages.front() + ".in1");
        }
        if (const std::optional<uint32_t> writer = in_place_stage()) {
            for (const std::string port : {"inp", "binp"}) {
                subgraph.add_connection("." + port, stages[*writer] + "." + port);
                subgraph.add_connection(stages[*writer] + "." + port, "." + port);
            }
        }
        if (settings.broken == static_cast<uint32_t>(Broken::MISSING_INPUT)) {
            subgraph.add_connection(".in2", stages.front() + ".missing");
        } else if (settings.broken == static_cast<uint32_t>(Broken::UNKNOWN_NODE)) {
            subgraph.add_connection(".in2", "unknown.in0");
        }
        return subgraph;
    }

    std::vector<InputConnectorDescriptor> describe_inputs() override {
        if (settings.own_input.empty()) {
            return {};
        }
        return {{.name = settings.own_input,
                 .connector = con_own,
                 .access = ConnectorAccess::transfer_src,
                 .delay = settings.own_delay,
                 .optional = !settings.own_required}};
    }

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        if (!settings.own_output) {
            return {};
        }
        return {{"out", ManagedVkImageOut::create(vk::Format::eR32Uint, 16, 16),
                 ConnectorAccess::transfer_dst}};
    }

    std::optional<uint32_t> in_place_stage() const {
        for (uint32_t i = 0; i < children.size(); i++) {
            if (children[i]->shape.in_place) {
                return i;
            }
        }
        if (inner && inner->in_place_stage()) {
            return static_cast<uint32_t>(children.size());
        }
        return std::nullopt;
    }

    NodeStatusFlags pre_process(const NodeIO& /*io*/, const NodeProcessInfo& /*info*/) override {
        if (pending_properties) {
            JSONLoadProperties props(*pending_properties);
            pending_properties.reset();
            std::ignore = properties(props);
            return NEEDS_RECONNECT;
        }
        return {};
    }

    NodeStatusFlags properties(Properties& props) override {
        auto count = static_cast<uint32_t>(children.size());
        bool with_inner = inner != nullptr;
        Settings new_settings = settings;
        props.config_uint("children", count);
        props.config_bool("inner group", with_inner);
        std::ignore = props.config_text("prefix", new_settings.prefix);
        props.config_uint("broken", new_settings.broken);
        props.config_bool("parallel", new_settings.parallel);
        props.config_bool("public in1", new_settings.public_in1);
        std::ignore = props.config_text("own input", new_settings.own_input);
        props.config_uint("own delay", new_settings.own_delay);
        props.config_bool("own required", new_settings.own_required);
        props.config_bool("own output", new_settings.own_output);
        NodeStatusFlags flags =
            count != children.size() || with_inner != (inner != nullptr) || new_settings != settings
                ? NEEDS_RECONNECT
                : 0;
        settings = new_settings;
        children.resize(count);
        for (std::shared_ptr<PatternNode>& child : children) {
            if (!child) {
                child = std::make_shared<PatternNode>();
            }
        }
        if (!with_inner) {
            inner.reset();
        } else if (!inner) {
            inner = std::make_shared<PatternGroup>();
        }

        for (uint32_t i = 0; i < children.size(); i++) {
            if (props.st_begin_child(fmt::format("child {}", i), fmt::format("Child {}", i))) {
                flags |= children[i]->properties(props);
                props.st_end_child();
            }
        }
        if (inner && props.st_begin_child("inner", "Inner")) {
            flags |= inner->properties(props);
            props.st_end_child();
        }
        return flags;
    }

    void set_properties(const json& properties) {
        pending_properties = properties;
    }

    std::vector<std::shared_ptr<PatternNode>> children;
    std::shared_ptr<PatternGroup> inner;

  private:
    Settings settings;
    std::optional<json> pending_properties;
    const VkImageInHandle con_own = VkImageIn::create();
};

class SwapGroup : public Node {
  public:
    SwapGroup() {
        for (const std::shared_ptr<PatternGroup>& group : groups) {
            group->children = {std::make_shared<PatternNode>(), std::make_shared<PatternNode>()};
        }
    }

    Subgraph describe_subgraph() override {
        Subgraph subgraph;
        if (swapped) {
            subgraph.add_node("a", groups[1]);
            subgraph.add_node("b", groups[0]);
        } else {
            subgraph.add_node("a", groups[0]);
            subgraph.add_node("b", groups[1]);
        }
        for (const std::string name : {"a", "b"}) {
            subgraph.add_connection(".in0", name + ".in0");
        }
        subgraph.add_connection("a.out", ".out");
        return subgraph;
    }

    NodeStatusFlags properties(Properties& props) override {
        const bool swapped_changed = props.config_bool("swapped", swapped);
        if (!props.config_bool("fail", fail) && !swapped_changed) {
            return {};
        }
        groups[0]->children.resize(swapped ? 1 : 2);
        if (!groups[0]->children.back()) {
            groups[0]->children.back() = std::make_shared<PatternNode>();
        }
        const PatternGroup::Broken broken =
            swapped && fail ? PatternGroup::Broken::THROWS : PatternGroup::Broken::NO;
        JSONLoadProperties broken_props(json{{"broken", static_cast<uint32_t>(broken)}});
        std::ignore = groups[0]->properties(broken_props);
        return NEEDS_RECONNECT;
    }

    const std::array<std::shared_ptr<PatternGroup>, 2> groups = {std::make_shared<PatternGroup>(),
                                                                 std::make_shared<PatternGroup>()};

  private:
    bool swapped = false;
    bool fail = false;
};

class MoveGroup : public Node {
  public:
    enum class Place : uint32_t {
        A,
        B,
        HERE,
    };

    MoveGroup() {
        for (const std::shared_ptr<PatternGroup>& group : groups) {
            group->children = {std::make_shared<PatternNode>()};
        }
        groups[0]->children.push_back(x);
    }

    Subgraph describe_subgraph() override {
        Subgraph subgraph;
        subgraph.add_node("a", groups[0]);
        subgraph.add_node("b", groups[1]);
        subgraph.add_connection(".in0", "a.in0");
        subgraph.add_connection(".in0", "b.in0");
        subgraph.add_connection("a.out", ".out");
        if (place == static_cast<uint32_t>(Place::HERE)) {
            subgraph.add_node("x", x);
            subgraph.add_connection(".in0", "x.in0");
        }
        return subgraph;
    }

    NodeStatusFlags properties(Properties& props) override {
        const bool replacing = props.config_bool("replace", replace);
        if (!props.config_uint("place", place) && !replacing) {
            return {};
        }
        for (uint32_t index = 0; index < groups.size(); index++) {
            std::erase(groups[index]->children, x);
            std::erase(groups[index]->children, stand_ins[index]);
            if (index == place) {
                groups[index]->children.push_back(x);
            } else if (replace) {
                groups[index]->children.push_back(stand_ins[index]);
            }
        }
        return NEEDS_RECONNECT;
    }

    const std::array<std::shared_ptr<PatternGroup>, 2> groups = {std::make_shared<PatternGroup>(),
                                                                 std::make_shared<PatternGroup>()};
    const std::shared_ptr<PatternNode> x = std::make_shared<PatternNode>();
    // take the place of x in the group it is not in
    const std::array<std::shared_ptr<PatternNode>, 2> stand_ins = {std::make_shared<PatternNode>(),
                                                                   std::make_shared<PatternNode>()};
    bool replace = false;

  private:
    uint32_t place = static_cast<uint32_t>(Place::A);
};

class UnregisteredNode : public Node {};

class ThrowingNode : public Node {
  public:
    void initialize(const ContextHandle& /*context*/,
                    const ResourceAllocatorHandle& /*allocator*/) override {
        throw std::runtime_error{"failing on purpose"};
    }
};

class ScriptedGroup : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return inputs;
    }

    Subgraph describe_subgraph() override {
        return script(*this);
    }

    std::function<Subgraph(ScriptedGroup&)> script = [](ScriptedGroup&) { return Subgraph{}; };
    std::vector<InputConnectorDescriptor> inputs;
    const std::array<std::shared_ptr<PatternNode>, 2> children = {std::make_shared<PatternNode>(),
                                                                  std::make_shared<PatternNode>()};
};

class UnsupportedPatternGroup : public PatternGroup {
  public:
    void initialize(const ContextHandle& /*context*/,
                    const ResourceAllocatorHandle& /*allocator*/) override {
        throw graph_errors::node_error{"not on this device"};
    }
};

class IntSource : public Node {
  public:
    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        return {{"out", con_out}};
    }

    NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& /*submission*/) override {
        io[con_out] = std::make_shared<int>(static_cast<int>(info.get_total_iteration()));
        return {};
    }

  private:
    const PtrOutHandle<int> con_out = PtrOut<int>::create();
};

class FixedIntSource : public Node {
  public:
    class FixedOut : public PtrOut<int> {
      public:
        FixedOut() : PtrOut<int>(false) {}

        bool supports_in_place() const override {
            return false;
        }
    };

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        return {{"out", con_out}};
    }

    NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& /*submission*/) override {
        io[con_out] = std::make_shared<int>(static_cast<int>(info.get_total_iteration()));
        return {};
    }

  private:
    const std::shared_ptr<FixedOut> con_out = std::make_shared<FixedOut>();
};

class IntAdd : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {{.name = "inp", .connector = con_in_place, .in_place = true}};
    }

    NodeStatusFlags process(const NodeIO& io,
                            const NodeProcessInfo& /*info*/,
                            Submission& /*submission*/) override {
        *io[con_in_place] += 1000;
        return {};
    }

  private:
    const PtrInHandle<int> con_in_place = PtrIn<int>::create();
};

class IntReader : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {{.name = "in", .connector = con_in}};
    }

    NodeStatusFlags process(const NodeIO& io,
                            const NodeProcessInfo& /*info*/,
                            Submission& /*submission*/) override {
        values.push_back(*io[con_in]);
        return {};
    }

    std::vector<int> values;

  private:
    const PtrInHandle<int> con_in = PtrIn<int>::create();
};
