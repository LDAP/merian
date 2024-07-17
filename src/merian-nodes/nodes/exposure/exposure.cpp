#include "exposure.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian_nodes {

static const uint32_t histogram_spv[] = {
#include "histogram.comp.spv.h"
};

static const uint32_t luminance_spv[] = {
#include "luminance.comp.spv.h"
};

static const uint32_t exposure_spv[] = {
#include "exposure.comp.spv.h"
};

AutoExposure::AutoExposure(const ContextHandle context) : Node(), context(context) {

    histogram_module =
        std::make_shared<ShaderModule>(context, sizeof(histogram_spv), histogram_spv);
    luminance_module =
        std::make_shared<ShaderModule>(context, sizeof(luminance_spv), luminance_spv);
    exposure_module = std::make_shared<ShaderModule>(context, sizeof(exposure_spv), exposure_spv);
}

AutoExposure::~AutoExposure() {}

std::vector<InputConnectorHandle> AutoExposure::describe_inputs() {
    return {con_src};
}

std::vector<OutputConnectorHandle>
AutoExposure::describe_outputs(const ConnectorIOMap& output_for_input) {
    const vk::Format format = output_for_input[con_src]->create_info.format;
    const vk::Extent3D extent = output_for_input[con_src]->create_info.extent;

    con_out = ManagedVkImageOut::compute_write("out", format, extent);
    con_hist = std::make_shared<ManagedVkBufferOut>(
        "histogram",
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
            vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo({}, local_size_x * local_size_y * sizeof(uint32_t) + sizeof(uint32_t),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst));
    con_luminance = std::make_shared<ManagedVkBufferOut>(
        "avg_luminance", vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader, vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo({}, sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer), true);

    return {con_out, con_hist, con_luminance};
}

AutoExposure::NodeStatusFlags
AutoExposure::on_connected(const DescriptorSetLayoutHandle& descriptor_set_layout) {
    if (!exposure) {
        auto pipe_layout = PipelineLayoutBuilder(context)
                               .add_descriptor_set_layout(descriptor_set_layout)
                               .add_push_constant<PushConstant>()
                               .build_pipeline_layout();
        auto spec_builder = SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y);
        SpecializationInfoHandle spec = spec_builder.build();

        histogram = std::make_shared<ComputePipeline>(pipe_layout, histogram_module, spec);
        luminance = std::make_shared<ComputePipeline>(pipe_layout, luminance_module, spec);
        exposure = std::make_shared<ComputePipeline>(pipe_layout, exposure_module, spec);
    }

    return {};
}

void AutoExposure::process(GraphRun& run,
                           const vk::CommandBuffer& cmd,
                           const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {
    const auto group_count_x = (io[con_out]->get_extent().width + local_size_x - 1) / local_size_x;
    const auto group_count_y = (io[con_out]->get_extent().height + local_size_y - 1) / local_size_y;

    if (pc.automatic) {
        pc.reset = run.get_iteration() == 0;
        pc.timediff = sw.seconds();
        sw.reset();

        auto bar = io[con_hist]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                                    vk::AccessFlagBits::eShaderWrite,
                                                vk::AccessFlagBits::eTransferWrite);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer, {}, {}, bar, {});
        io[con_hist]->fill(cmd);
        bar = io[con_hist]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                           vk::AccessFlagBits::eShaderRead |
                                               vk::AccessFlagBits::eShaderWrite);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, bar, {});

        histogram->bind(cmd);
        histogram->bind_descriptor_set(cmd, descriptor_set);
        histogram->push_constant(cmd, pc);
        cmd.dispatch(group_count_x, group_count_y, 1);

        bar = io[con_hist]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                               vk::AccessFlagBits::eShaderWrite,
                                           vk::AccessFlagBits::eShaderRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, bar, {});
    }

    luminance->bind(cmd);
    luminance->bind_descriptor_set(cmd, descriptor_set);
    luminance->push_constant(cmd, pc);
    cmd.dispatch(1, 1, 1);

    auto bar = io[con_luminance]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                                     vk::AccessFlagBits::eShaderWrite,
                                                 vk::AccessFlagBits::eShaderRead);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, bar, {});

    exposure->bind(cmd);
    exposure->bind_descriptor_set(cmd, descriptor_set);
    exposure->push_constant(cmd, pc);
    cmd.dispatch(group_count_x, group_count_y, 1);
}

AutoExposure::NodeStatusFlags AutoExposure::properties(Properties& config) {
    config.st_separate("General");
    bool autoexposure = pc.automatic;
    config.config_bool("autoexposure", autoexposure);
    pc.automatic = autoexposure;
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
    float shutter_time = pc.shutter_time * 1000.0;
    config.config_float("shutter time (ms)", shutter_time);
    pc.shutter_time = std::max(0.f, shutter_time / 1000);
    config.config_float("aperature", pc.aperature, "", .01);

    return {};
}

} // namespace merian_nodes
