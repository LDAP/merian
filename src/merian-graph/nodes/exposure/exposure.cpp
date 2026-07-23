#include "merian-graph/nodes/exposure/exposure.hpp"

#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

namespace {
constexpr const char* HISTOGRAM_MODULE = "merian-graph/nodes/exposure/histogram.slang";
constexpr const char* LUMINANCE_MODULE = "merian-graph/nodes/exposure/luminance.slang";
constexpr const char* EXPOSURE_MODULE = "merian-graph/nodes/exposure/exposure.slang";
} // namespace

AutoExposure::AutoExposure() {}

AutoExposure::~AutoExposure() {}

DeviceSupportInfo AutoExposure::query_device_support(const DeviceSupportQueryInfo& query_info) {
    DeviceSupportInfo support{true};
    for (const char* module : {HISTOGRAM_MODULE, LUMINANCE_MODULE, EXPOSURE_MODULE}) {
        const auto composition = SlangComposition::create();
        composition->add_module_from_path(module, true);
        support = support & SlangProgram::create(query_info.compile_context, composition)
                                .get()
                                ->query_device_support(query_info);
    }
    return support;
}

void AutoExposure::initialize(const ContextHandle& context,
                              const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(LOCAL_SIZE_X, LOCAL_SIZE_Y);
    spec_info.set(spec_builder.build());

    histogram_kernel.emplace(context, allocator, compile_context, HISTOGRAM_MODULE, spec_info);
    luminance_kernel.emplace(context, allocator, compile_context, LUMINANCE_MODULE, spec_info);
    exposure_kernel.emplace(context, allocator, compile_context, EXPOSURE_MODULE, spec_info);
}

std::vector<InputConnectorDescriptor> AutoExposure::describe_inputs() {
    return {{"src", con_src, ConnectorAccess::compute_read}};
}

std::vector<OutputConnectorDescriptor>
AutoExposure::describe_outputs(const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_src]->get_create_info_or_throw();
    const vk::Format format = create_info.format;
    const vk::Extent3D extent = create_info.extent;

    con_out = ManagedVkImageOut::create(format, extent);
    con_hist = ManagedVkBufferOut::create(vk::BufferCreateInfo(
        {}, ((vk::DeviceSize)LOCAL_SIZE_X * LOCAL_SIZE_Y * sizeof(uint32_t)) + sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst));
    con_luminance = ManagedVkBufferOut::create(
        vk::BufferCreateInfo({}, sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer), true);

    return {{"out", con_out, ConnectorAccess::compute_write},
            {"histogram", con_hist,
             ConnectorAccess::compute_read_write | ConnectorAccess::transfer_dst},
            {"avg_luminance", con_luminance, ConnectorAccess::compute_read_write}};
}

AutoExposure::NodeStatusFlags AutoExposure::on_connected(const NodeConnectedInfo& info) {
    const NodeIOLayout& io_layout = info.io_layout;
    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            for (auto* kernel : {&histogram_kernel, &luminance_kernel, &exposure_kernel}) {
                (*kernel)->reload(std::any_cast<bool>(force), compile_context);
            }
            return true;
        });

    return {};
}

void AutoExposure::process(GraphRun& run, const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();
    if (pc.automatic == VK_TRUE) {
        pc.reset = run.get_iteration() == 0 ? VK_TRUE : VK_FALSE;
        pc.timediff = static_cast<float>(run.get_time_delta());

        auto bar = io[con_hist]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                                    vk::AccessFlagBits::eShaderWrite,
                                                vk::AccessFlagBits::eTransferWrite);
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eTransfer, bar);

        cmd->fill(io[con_hist]);

        bar = io[con_hist]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                           vk::AccessFlagBits::eShaderRead |
                                               vk::AccessFlagBits::eShaderWrite);
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eComputeShader, bar);

        const auto pipe = histogram_kernel->bind(run, io);
        cmd->push_constant(pipe, pc);
        cmd->dispatch(io[con_out]->get_extent(), LOCAL_SIZE_X, LOCAL_SIZE_Y);

        bar = io[con_hist]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                               vk::AccessFlagBits::eShaderWrite,
                                           vk::AccessFlagBits::eShaderRead);
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader, bar);
    }

    const auto luminance_pipe = luminance_kernel->bind(run, io);
    cmd->push_constant(luminance_pipe, pc);
    cmd->dispatch(1, 1, 1);

    auto bar = io[con_luminance]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                                     vk::AccessFlagBits::eShaderWrite,
                                                 vk::AccessFlagBits::eShaderRead);
    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                 vk::PipelineStageFlagBits::eComputeShader, bar);

    const auto exposure_pipe = exposure_kernel->bind(run, io);
    cmd->push_constant(exposure_pipe, pc);
    cmd->dispatch(io[con_out]->get_extent(), LOCAL_SIZE_X, LOCAL_SIZE_Y);
}

AutoExposure::NodeStatusFlags AutoExposure::properties(Properties& config) {
    config.st_separate("General");
    config.config_bool("autoexposure", pc.automatic);
    config.config_float("q", pc.q, "Lens and vignetting attenuation", 0.01);
    config.config_float("min exposure", pc.min_exposure, "", 0.1);
    pc.min_exposure = std::max(pc.min_exposure, 0.f);
    pc.max_exposure = std::max(pc.max_exposure, pc.min_exposure);
    config.config_float("max exposure", pc.max_exposure, "", 0.1);
    pc.max_exposure = std::max(pc.max_exposure, 0.f);
    pc.min_exposure = std::min(pc.min_exposure, pc.max_exposure);

    config.st_separate("Auto");
    config.config_float("K", pc.K, "Reflected-light meter calibration constant", 0.1);
    config.config_float("min log luminance", pc.min_log_histogram);
    config.config_float("max log luminance", pc.max_log_histogram);
    config.config_float("speed up", pc.speed_up);
    config.config_float("speed down", pc.speed_down);
    config.config_options("metering", pc.metering, {"uniform", "center-weighted", "center"},
                          Properties::OptionsStyle::COMBO);

    config.st_separate("Manual");
    config.config_float("ISO", pc.iso, "Sensor sensitivity/gain (ISO)");
    float shutter_time = pc.shutter_time * 1000.0f;
    config.config_float("shutter time (ms)", shutter_time);
    pc.shutter_time = std::max(0.f, shutter_time / 1000);
    config.config_float("aperature", pc.aperature, "", .01);

    return {};
}

} // namespace merian
