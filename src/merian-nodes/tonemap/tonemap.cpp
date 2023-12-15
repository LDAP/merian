#include "tonemap.hpp"
#include "config.h"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

static const uint32_t spv[] = {
#include "tonemap.comp.spv.h"
};

namespace merian {

TonemapNode::TonemapNode(const SharedContext context,
                         const ResourceAllocatorHandle allocator,
                         const std::optional<vk::Format> output_format)
    : ComputeNode(context, allocator, sizeof(PushConstant)), output_format(output_format) {
    shader = std::make_shared<ShaderModule>(context, sizeof(spv), spv);
}

TonemapNode::~TonemapNode() {}

std::string TonemapNode::name() {
    return "Tonemap";
}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
TonemapNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("src"),
        },
        {},
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
TonemapNode::describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                              const std::vector<NodeOutputDescriptorBuffer>&) {
    extent = connected_image_outputs[0].create_info.extent;
    vk::Format format = output_format.value_or(connected_image_outputs[0].create_info.format);

    return {
        {
            NodeOutputDescriptorImage::compute_write("output", format, extent),
        },
        {},
    };
}

SpecializationInfoHandle TonemapNode::get_specialization_info() const noexcept {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, tonemap, alpha_mode, clamp_output);
    return spec_builder.build();
}

const void* TonemapNode::get_push_constant([[maybe_unused]] GraphRun& run) {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t> TonemapNode::get_group_count() const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle TonemapNode::get_shader_module() {
    return shader;
}

void TonemapNode::get_configuration(Configuration& config, bool& needs_rebuild) {
    const int old_tonemap = tonemap;
    config.config_options(
        "tonemap", tonemap,
        {"None", "Clamp", "Uncharted 2", "Reinhard Extended", "Aces", "Aces-Approx", "Lottes"});
    needs_rebuild |= old_tonemap != tonemap;

    if (tonemap == TONEMAP_REINHARD_EXTENDED) {
        if (old_tonemap != TONEMAP_REINHARD_EXTENDED)
            pc.param1 = 1.0;
        config.config_float("max white", pc.param1, "max luminance found in the scene", .05);
    }

    if (tonemap == TONEMAP_UNCHARTED_2) {
        if (old_tonemap != TONEMAP_UNCHARTED_2) {
            pc.param1 = 2.0;
            pc.param2 = 11.2;
        }
        config.config_float("exposure bias", pc.param1, "see UNCHARTED 2", .05);
        config.config_float("W", pc.param2, "see UNCHARTED 2", .1);
    }

    if (tonemap == TONEMAP_LOTTES) {
        if (old_tonemap != TONEMAP_LOTTES) {
            pc.param1 = 1.0;
            pc.param2 = 1.0;
            pc.param3 = 16.0;
            pc.param4 = 0.18;
            pc.param5 = 0.18;
        }
        config.config_float("contrast", pc.param1, "See Lottes talk", 0.01);
        config.config_float("shoulder", pc.param2, "See Lottes talk", 0.01);
        config.config_float("hdrMax", pc.param3, "See Lottes talk", 0.1);
        config.config_float("midIn", pc.param4, "See Lottes talk", 0.001);
        config.config_float("midOut", pc.param5, "See Lottes talk", 0.001);
    }

    config.st_separate();
    int32_t old_clamp_output = clamp_output;
    config.config_bool("clamp output", clamp_output, "clamps the output (before computing the alpha channel)");
    needs_rebuild |= old_clamp_output != clamp_output;

    config.st_separate();
    int32_t old_alpha_mode = alpha_mode;
    config.config_options("alpha mode", alpha_mode,
                          {
                              "Passthrough",
                              "Luminance",
                              "Perceptual luminance",
                          },
                          Configuration::OptionsStyle::DONT_CARE,
                          "Decides what is written in the alpha channel.");
    if (alpha_mode == ALPHA_MODE_PERCEPTUAL_LUMINANCE) {
        config.config_float(
            "perceptual exponent", pc.perceptual_exponent,
            "Adjust the exponent that is used to convert the luminance to perceptual space.", 0.1);
    }

    needs_rebuild |= old_alpha_mode != alpha_mode;
}

} // namespace merian
