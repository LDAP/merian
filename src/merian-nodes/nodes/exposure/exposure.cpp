#include "merian-nodes/nodes/exposure/exposure.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "exposure.slang.spv.h"
#include "histogram.slang.spv.h"
#include "luminance.slang.spv.h"

namespace merian_nodes {

AutoExposure::AutoExposure(const ContextHandle& context) : Node(), context(context) {

    histogram_module = ShaderModule::create(
        context, merian_histogram_slang_spv(), merian_histogram_slang_spv_size(),
        ShaderModule::EntryPointInfo("main", vk::ShaderStageFlagBits::eCompute));
    luminance_module = ShaderModule::create(
        context, merian_luminance_slang_spv(), merian_luminance_slang_spv_size(),
        ShaderModule::EntryPointInfo("main", vk::ShaderStageFlagBits::eCompute));
    exposure_module = ShaderModule::create(
        context, merian_exposure_slang_spv(), merian_exposure_slang_spv_size(),
        ShaderModule::EntryPointInfo("main", vk::ShaderStageFlagBits::eCompute));
}

AutoExposure::~AutoExposure() {}

std::vector<InputConnectorHandle> AutoExposure::describe_inputs() {
    return {con_src};
}

std::vector<OutputConnectorHandle> AutoExposure::describe_outputs(const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_src]->get_create_info_or_throw();
    const vk::Format format = create_info.format;
    const vk::Extent3D extent = create_info.extent;

    con_out = ManagedVkImageOut::compute_write("out", format, extent);
    con_hist = std::make_shared<ManagedVkBufferOut>(
        "histogram",
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
            vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo(
            {}, ((vk::DeviceSize)LOCAL_SIZE_X * LOCAL_SIZE_Y * sizeof(uint32_t)) + sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst));
    con_luminance = std::make_shared<ManagedVkBufferOut>(
        "avg_luminance", vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader, vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo({}, sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer), true);

    return {con_out, con_hist, con_luminance};
}

AutoExposure::NodeStatusFlags
AutoExposure::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                           const DescriptorSetLayoutHandle& descriptor_set_layout) {
    if (!exposure) {
        auto pipe_layout = PipelineLayoutBuilder(context)
                               .add_descriptor_set_layout(descriptor_set_layout)
                               .add_push_constant<PushConstant>()
                               .build_pipeline_layout();
        auto spec_builder = SpecializationInfoBuilder();
        spec_builder.add_entry(LOCAL_SIZE_X, LOCAL_SIZE_Y);
        SpecializationInfoHandle spec = spec_builder.build();

        histogram = std::make_shared<ComputePipeline>(pipe_layout, histogram_module, spec);
        luminance = std::make_shared<ComputePipeline>(pipe_layout, luminance_module, spec);
        exposure = std::make_shared<ComputePipeline>(pipe_layout, exposure_module, spec);
    }

    return {};
}

void AutoExposure::process(GraphRun& run,
                           const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {
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

        cmd->bind(histogram);
        cmd->bind_descriptor_set(histogram, descriptor_set);
        cmd->push_constant(histogram, pc);
        cmd->dispatch(io[con_out]->get_extent(), LOCAL_SIZE_X, LOCAL_SIZE_Y);

        bar = io[con_hist]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                               vk::AccessFlagBits::eShaderWrite,
                                           vk::AccessFlagBits::eShaderRead);
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader, bar);
    }

    cmd->bind(luminance);
    cmd->bind_descriptor_set(luminance, descriptor_set);
    cmd->push_constant(luminance, pc);
    cmd->dispatch(1, 1, 1);

    auto bar = io[con_luminance]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                                     vk::AccessFlagBits::eShaderWrite,
                                                 vk::AccessFlagBits::eShaderRead);
    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                 vk::PipelineStageFlagBits::eComputeShader, bar);

    cmd->bind(exposure);
    cmd->bind_descriptor_set(exposure, descriptor_set);
    cmd->push_constant(exposure, pc);
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

} // namespace merian_nodes
