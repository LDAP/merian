#include "merian-graph/nodes/bloom/bloom.hpp"

#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

namespace {
constexpr const char* SEPARATE_MODULE = "merian-graph/nodes/bloom/bloom_separate.slang";
constexpr const char* COMPOSITE_MODULE = "merian-graph/nodes/bloom/bloom_composite.slang";
} // namespace

Bloom::Bloom() {}

Bloom::~Bloom() {}

DeviceSupportInfo Bloom::query_device_support(const DeviceSupportQueryInfo& query_info) {
    DeviceSupportInfo support{true};
    for (const char* module : {SEPARATE_MODULE, COMPOSITE_MODULE}) {
        const auto composition = SlangComposition::create();
        composition->add_module_from_path(module, true);
        support = support & SlangProgram::create(query_info.compile_context, composition)
                                .get()
                                ->query_device_support(query_info);
    }
    return support;
}

void Bloom::initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, mode);
    spec_info.set(spec_builder.build());

    separate_kernel.emplace(context, allocator, compile_context, SEPARATE_MODULE, spec_info);
    composite_kernel.emplace(context, allocator, compile_context, COMPOSITE_MODULE, spec_info);
}

std::vector<InputConnectorDescriptor> Bloom::describe_inputs() {
    return {{"src", con_src, ConnectorAccess::compute_read}};
}

std::vector<OutputConnectorDescriptor> Bloom::describe_outputs(const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_src]->get_create_info_or_throw();
    const vk::Format format = create_info.format;
    const vk::Extent3D extent = create_info.extent;

    con_out = ManagedVkImageOut::create(format, extent);
    con_interm = ManagedVkImageOut::create(vk::Format::eR16G16B16A16Sfloat, extent);

    return {
        {"out", con_out, ConnectorAccess::compute_write},
        {"interm", con_interm, ConnectorAccess::compute_read_write},
    };
}

Bloom::NodeStatusFlags Bloom::on_connected(const NodeConnectedInfo& info) {
    const NodeIOLayout& io_layout = info.io_layout;
    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            for (auto* kernel : {&separate_kernel, &composite_kernel}) {
                (*kernel)->reload(std::any_cast<bool>(force), compile_context);
            }
            return true;
        });

    return {};
}

void Bloom::process(GraphRun& run, const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();
    const auto separate_pipe = separate_kernel->bind(run, io);
    cmd->push_constant(separate_pipe, pc);
    cmd->dispatch(io[con_out]->get_extent(), local_size_x, local_size_y);

    const auto bar =
        io[con_interm]->barrier(vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eShaderRead);
    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                 vk::PipelineStageFlagBits::eComputeShader, bar);

    const auto composite_pipe = composite_kernel->bind(run, io);
    cmd->push_constant(composite_pipe, pc);
    cmd->dispatch(io[con_out]->get_extent(), local_size_x, local_size_y);
}

Bloom::NodeStatusFlags Bloom::properties(Properties& config) {
    config.config_float("brightness threshold", pc.threshold,
                        "Only areas brighter than that are affected", .1);
    config.config_float("strengh", pc.strength, "Controls the strength of the effect", .0001);

    config.st_separate("Debug");
    const bool value_changed =
        config.config_options("mode", mode, {"combined", "bloom only", "bloom off"});

    if (value_changed) {
        auto spec_builder = SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y, mode);
        spec_info.set(spec_builder.build());
    }
    return {};
}

} // namespace merian
