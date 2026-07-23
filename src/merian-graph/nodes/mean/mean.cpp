#include "merian-graph/nodes/mean/mean.hpp"

#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

namespace {
constexpr const char* IMAGE_TO_BUFFER_MODULE = "merian-graph/nodes/mean/image_to_buffer.slang";
constexpr const char* REDUCE_BUFFER_MODULE = "merian-graph/nodes/mean/reduce_buffer.slang";
} // namespace

MeanToBuffer::MeanToBuffer() {}

MeanToBuffer::~MeanToBuffer() {}

DeviceSupportInfo MeanToBuffer::query_device_support(const DeviceSupportQueryInfo& query_info) {
    DeviceSupportInfo support{true};
    for (const char* module : {IMAGE_TO_BUFFER_MODULE, REDUCE_BUFFER_MODULE}) {
        const auto composition = SlangComposition::create();
        composition->add_module_from_path(module, true);
        support = support & SlangProgram::create(query_info.compile_context, composition)
                                .get()
                                ->query_device_support(query_info);
    }
    return support;
}

void MeanToBuffer::initialize(const ContextHandle& context,
                              const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    auto i2b_spec = SpecializationInfoBuilder();
    i2b_spec.add_entry(local_size_x, local_size_y);
    image_to_buffer_spec.set(i2b_spec.build());

    auto reduce_spec = SpecializationInfoBuilder();
    reduce_spec.add_entry(workgroup_size, 1u);
    reduce_buffer_spec.set(reduce_spec.build());

    image_to_buffer_kernel.emplace(context, allocator, compile_context, IMAGE_TO_BUFFER_MODULE,
                                   image_to_buffer_spec);
    reduce_buffer_kernel.emplace(context, allocator, compile_context, REDUCE_BUFFER_MODULE,
                                 reduce_buffer_spec);
}

std::vector<InputConnectorDescriptor> MeanToBuffer::describe_inputs() {
    return {{"src", con_src, ConnectorAccess::compute_read}};
}

std::vector<OutputConnectorDescriptor>
MeanToBuffer::describe_outputs(const NodeIOLayout& io_layout) {
    vk::Extent3D extent = io_layout[con_src]->get_create_info_or_throw().extent;

    const auto group_count_x = (extent.width + local_size_x - 1) / local_size_x;
    const auto group_count_y = (extent.height + local_size_y - 1) / local_size_y;
    const std::size_t buffer_size = group_count_x * group_count_y;

    con_mean = ManagedVkBufferOut::create(vk::BufferCreateInfo(
        {}, buffer_size * sizeof(merian::float4), vk::BufferUsageFlagBits::eStorageBuffer));

    return {{"mean", con_mean, ConnectorAccess::compute_read_write}};
}

MeanToBuffer::NodeStatusFlags MeanToBuffer::on_connected(const NodeConnectedInfo& info) {
    const NodeIOLayout& io_layout = info.io_layout;
    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            for (auto* kernel : {&image_to_buffer_kernel, &reduce_buffer_kernel}) {
                (*kernel)->reload(std::any_cast<bool>(force), compile_context);
            }
            return true;
        });

    return {};
}

void MeanToBuffer::process([[maybe_unused]] GraphRun& run, const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();
    const auto group_count_x = (io[con_src]->get_extent().width + local_size_x - 1) / local_size_x;
    const auto group_count_y = (io[con_src]->get_extent().height + local_size_y - 1) / local_size_y;

    pc.divisor = io[con_src]->get_extent().width * io[con_src]->get_extent().height;

    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "image to buffer");
        const auto pipe = image_to_buffer_kernel->bind(run, io);
        cmd->push_constant(pipe, pc);
        cmd->dispatch(group_count_x, group_count_y, 1);
    }

    pc.size = group_count_x * group_count_y;
    pc.offset = 1;
    pc.count = group_count_x * group_count_y;

    PipelineHandle reduce_pipe;
    while (pc.count > 1) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd,
                                 fmt::format("reduce {} elements", pc.count));
        auto bar = io[con_mean]->buffer_barrier(
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader, bar);

        if (!reduce_pipe) {
            reduce_pipe = reduce_buffer_kernel->bind(run, io);
        }
        cmd->push_constant(reduce_pipe, pc);
        cmd->dispatch((pc.count + workgroup_size - 1) / workgroup_size, 1, 1);

        pc.count = (pc.count + workgroup_size - 1) / workgroup_size;
        pc.offset *= workgroup_size;
    }
}

} // namespace merian
