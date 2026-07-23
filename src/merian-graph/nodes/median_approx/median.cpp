#include "merian-graph/nodes/median_approx/median.hpp"

#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

namespace {
constexpr const char* HISTOGRAM_MODULE = "merian-graph/nodes/median_approx/median_histogram.slang";
constexpr const char* REDUCE_MODULE = "merian-graph/nodes/median_approx/median_reduce.slang";
} // namespace

MedianApproxNode::MedianApproxNode() {}

MedianApproxNode::~MedianApproxNode() {}

DeviceSupportInfo MedianApproxNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    DeviceSupportInfo support{true};
    for (const char* module : {HISTOGRAM_MODULE, REDUCE_MODULE}) {
        const auto composition = SlangComposition::create();
        composition->add_module_from_path(module, true);
        support = support & SlangProgram::create(query_info.compile_context, composition)
                                .get()
                                ->query_device_support(query_info);
    }
    return support;
}

void MedianApproxNode::initialize(const ContextHandle& context,
                                  const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
    this->compile_context = context->get_shader_compile_context();
    make_spec_info();

    histogram_kernel.emplace(context, allocator, compile_context, HISTOGRAM_MODULE, spec_info);
    reduce_kernel.emplace(context, allocator, compile_context, REDUCE_MODULE, spec_info);
}

void MedianApproxNode::make_spec_info() {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, component);
    spec_info.set(spec_builder.build());
}

std::vector<InputConnectorDescriptor> MedianApproxNode::describe_inputs() {
    return {{"src", con_src, ConnectorAccess::compute_read}};
}

std::vector<OutputConnectorDescriptor>
MedianApproxNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {

    con_median = ManagedVkBufferOut::create(
        vk::BufferCreateInfo({}, sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer));
    con_histogram = ManagedVkBufferOut::create(vk::BufferCreateInfo(
        {}, local_size_x * local_size_y * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst));
    return {{"median", con_median, ConnectorAccess::compute_read_write},
            {"histogram", con_histogram,
             ConnectorAccess::compute_read_write | ConnectorAccess::transfer_dst}};
}

MedianApproxNode::NodeStatusFlags MedianApproxNode::on_connected(const NodeConnectedInfo& info) {
    const NodeIOLayout& io_layout = info.io_layout;
    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            for (auto* kernel : {&histogram_kernel, &reduce_kernel}) {
                (*kernel)->reload(std::any_cast<bool>(force), compile_context);
            }
            return true;
        });

    return {};
}

void MedianApproxNode::process([[maybe_unused]] GraphRun& run, [[maybe_unused]] const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();

    cmd->fill(io[con_histogram], 0);
    auto bar = io[con_histogram]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                 vk::AccessFlagBits::eShaderRead |
                                                     vk::AccessFlagBits::eShaderWrite);
    cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                 bar);

    const auto histogram_pipe = histogram_kernel->bind(run, io);
    cmd->push_constant(histogram_pipe, pc);
    cmd->dispatch(io[con_src]->get_extent(), local_size_x, local_size_y);

    bar = io[con_histogram]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                                vk::AccessFlagBits::eShaderWrite,
                                            vk::AccessFlagBits::eShaderRead);
    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                 vk::PipelineStageFlagBits::eComputeShader, bar);

    const auto reduce_pipe = reduce_kernel->bind(run, io);
    cmd->push_constant(reduce_pipe, pc);
    cmd->dispatch(1, 1, 1);
}

MedianApproxNode::NodeStatusFlags MedianApproxNode::properties(Properties& config) {
    if (config.config_options("component", component, {"R", "G", "B", "A"})) {
        make_spec_info();
    }

    config.config_float("min", pc.min);
    config.config_float("max", pc.max);

    return {};
}

} // namespace merian
