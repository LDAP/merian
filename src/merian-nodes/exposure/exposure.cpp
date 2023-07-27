#include "exposure.hpp"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/graph/node_utils.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

static const uint32_t histogram_spv[] = {
#include "histogram.comp.spv.h"
};

static const uint32_t luminance_spv[] = {
#include "luminance.comp.spv.h"
};

static const uint32_t exposure_spv[] = {
#include "exposure.comp.spv.h"
};

ExposureNode::ExposureNode(const SharedContext context, const ResourceAllocatorHandle allocator)
    : context(context), allocator(allocator) {

    histogram_module =
        std::make_shared<ShaderModule>(context, sizeof(histogram_spv), histogram_spv);
    luminance_module =
        std::make_shared<ShaderModule>(context, sizeof(luminance_spv), luminance_spv);
    exposure_module = std::make_shared<ShaderModule>(context, sizeof(exposure_spv), exposure_spv);
}

ExposureNode::~ExposureNode() {}

std::string ExposureNode::name() {
    return "Auto Exposure";
}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
ExposureNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("src"),
        },
        {},
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
ExposureNode::describe_outputs(
    const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
    const std::vector<NodeOutputDescriptorBuffer>&) {
    vk::Format format = connected_image_outputs[0].create_info.format;
    vk::Extent3D extent = connected_image_outputs[0].create_info.extent;
    return {
        {
            NodeOutputDescriptorImage::compute_write("output", format, extent),
        },
        {
            NodeOutputDescriptorBuffer(
                "histogram",
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderRead |
                    vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
                vk::BufferCreateInfo(
                    {}, local_size_x * local_size_y * sizeof(uint32_t) + sizeof(uint32_t),
                    vk::BufferUsageFlagBits::eStorageBuffer)),
            NodeOutputDescriptorBuffer(
                "avg_luminance",
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderRead,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::BufferCreateInfo({}, sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer),
                true),
        },
    };
}

void ExposureNode::cmd_build(const vk::CommandBuffer& cmd,
                             const std::vector<std::vector<ImageHandle>>& image_inputs,
                             const std::vector<std::vector<BufferHandle>>& buffer_inputs,
                             const std::vector<std::vector<ImageHandle>>& image_outputs,
                             const std::vector<std::vector<BufferHandle>>& buffer_outputs) {
    std::tie(graph_textures, graph_sets, graph_pool, graph_layout) =
        make_graph_descriptor_sets(context, allocator, image_inputs, buffer_inputs, image_outputs,
                                   buffer_outputs, graph_layout);

    if (!exposure) {
        auto pipe_layout = PipelineLayoutBuilder(context)
                               .add_descriptor_set_layout(graph_layout)
                               .add_push_constant<PushConstant>()
                               .build_pipeline_layout();
        auto spec_builder = SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y);
        SpecializationInfoHandle spec = spec_builder.build();

        histogram = std::make_shared<ComputePipeline>(pipe_layout, histogram_module, spec);
        luminance = std::make_shared<ComputePipeline>(pipe_layout, luminance_module, spec);
        exposure = std::make_shared<ComputePipeline>(pipe_layout, exposure_module, spec);
    }
}

void ExposureNode::cmd_process(const vk::CommandBuffer& cmd,
                               GraphRun& run,
                               const uint32_t set_index,
                               const std::vector<ImageHandle>& image_inputs,
                               const std::vector<BufferHandle>& buffer_inputs,
                               const std::vector<ImageHandle>& image_outputs,
                               const std::vector<BufferHandle>& buffer_outputs) {
    const auto group_count_x =
        (image_outputs[0]->get_extent().width + local_size_x - 1) / local_size_x;
    const auto group_count_y =
        (image_outputs[0]->get_extent().height + local_size_y - 1) / local_size_y;

    if (pc.automatic) {
        pc.reset = run.get_iteration() == 0;
        pc.timediff = sw.seconds();
        sw.reset();

        cmd.fillBuffer(*buffer_outputs[0], 0, VK_WHOLE_SIZE, 0);
        auto bar = buffer_outputs[0]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                     vk::AccessFlagBits::eShaderRead |
                                                         vk::AccessFlagBits::eShaderWrite);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, bar, {});

        histogram->bind(cmd);
        histogram->bind_descriptor_set(cmd, graph_sets[set_index]);
        histogram->push_constant(cmd, pc);
        cmd.dispatch(group_count_x, group_count_y, 1);
    }

    luminance->bind(cmd);
    luminance->bind_descriptor_set(cmd, graph_sets[set_index]);
    luminance->push_constant(cmd, pc);
    cmd.dispatch(1, 1, 1);

    auto bar = buffer_outputs[1]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                                     vk::AccessFlagBits::eShaderWrite,
                                                 vk::AccessFlagBits::eShaderRead);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, bar, {});

    exposure->bind(cmd);
    exposure->bind_descriptor_set(cmd, graph_sets[set_index]);
    exposure->push_constant(cmd, pc);
    cmd.dispatch(group_count_x, group_count_y, 1);
}

void ExposureNode::get_configuration(Configuration& config) {
    config.st_separate("General");
    bool autoexposure = pc.automatic;
    config.config_bool("autoexposure", autoexposure);
    pc.automatic = autoexposure;
    config.config_float("q", pc.q, "Lens and vignetting attenuation", 0.01);

    config.st_separate("Auto");
    config.config_float("K", pc.K, "Reflected-light meter calibration constant");
    config.config_float("min log luminance", pc.min_log_histogram);
    config.config_float("max log luminance", pc.max_log_histogram);
    config.config_float("speed up", pc.speed_up);
    config.config_float("speed down", pc.speed_down);
    config.config_options("metering", pc.metering, {"uniform", "center-weighted", "center"},
                          Configuration::OptionsStyle::COMBO);

    config.st_separate("Manual");
    config.config_float("ISO", pc.iso, "Sensor sensitivity/gain (ISO)");
    float shutter_time = pc.shutter_time * 1000.0;
    config.config_float("shutter time (ms)", shutter_time);
    pc.shutter_time = std::max(0.f, shutter_time / 1000);
    config.config_float("aperature", pc.aperature, "", .01);
}

} // namespace merian
