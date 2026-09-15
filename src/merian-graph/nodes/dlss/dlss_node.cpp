#include "merian-graph/nodes/dlss/dlss_node.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian/vk/utils/blits.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace merian {

namespace {
constexpr const char* AUTO_RESOLUTION_HINT = "0 outputs at the camera's resolution";

// The sequences that stay inside the +-0.5 px DLSS requires of the jitter (Programming Guide
// 3.7.3).
constexpr std::array<Camera::JitterSequence, 2> JITTER_SEQUENCES = {
    Camera::JitterSequence::Halton,
    Camera::JitterSequence::R2,
};
const std::vector<std::string> JITTER_SEQUENCE_NAMES = {"Halton", "R2"};

// indexed by DLSSNode::Mode
const std::vector<std::string> MODE_NAMES = {"bypass", "super resolution", "ray reconstruction"};

// Target over render resolution each mode is documented for, ordered by scale.
constexpr std::array<std::pair<DLSSQuality, float>, 5> QUALITY_SCALES = {{
    {DLSSQuality::DLAA, 1.f},
    {DLSSQuality::Quality, 1.5f},
    {DLSSQuality::Balanced, 1.724f},
    {DLSSQuality::Performance, 2.f},
    {DLSSQuality::UltraPerformance, 3.f},
}};

DLSSQuality quality_for_scale(const float scale) {
    // DLAA only where render and target resolution match
    if (std::abs(scale - 1.f) < 1e-3f) {
        return DLSSQuality::DLAA;
    }
    const auto closest = std::min_element(QUALITY_SCALES.begin() + 1, QUALITY_SCALES.end(),
                                          [scale](const auto& first, const auto& second) {
                                              return std::abs(first.second - scale) <
                                                     std::abs(second.second - scale);
                                          });
    return closest->first;
}

// "Total Phases = 8 * (target resolution / render resolution)^2", DLSS Programming Guide 3.7.1.1,
// floored at the 32 both guides prefer.
uint32_t recommended_jitter_phases(const float scale) {
    return std::max<uint32_t>(static_cast<uint32_t>(std::lround(8.f * scale * scale)), 32);
}
} // namespace

std::vector<std::string> DLSSNode::request_context_extensions() {
    return {ExtensionDLSSSuperResolution::name, ExtensionDLSSRayReconstruction::name};
}

DeviceSupportInfo DLSSNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    // ray reconstruction is checked on connect
    const auto super_resolution =
        query_info.extension_container.get_context_extension<ExtensionDLSSSuperResolution>(true);
    if (!super_resolution) {
        return {false, fmt::format("{} not available", ExtensionDLSSSuperResolution::name)};
    }
    return super_resolution->query_device_support(query_info);
}

void DLSSNode::initialize(const ContextHandle& context,
                          [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
    dlss_super_resolution = context->get_context_extension<ExtensionDLSSSuperResolution>();
    dlss_ray_reconstruction = context->get_context_extension<ExtensionDLSSRayReconstruction>(true);
}

std::vector<InputConnectorDescriptor> DLSSNode::describe_inputs() {
    switch (mode) {
    case Mode::Bypass:
        con_gbuffer = GBufferIn::create({});
        break;
    case Mode::SuperResolution:
        con_gbuffer = GBufferIn::create({
            {.fields = {GBufferField::ProjectedDepth}, .texture = true},
            {.fields = {GBufferField::MotionVectors}, .texture = true},
        });
        break;
    case Mode::RayReconstruction:
        con_gbuffer = GBufferIn::create({
            {.fields = {GBufferField::ViewDepth}, .texture = true},
            {.fields = {GBufferField::MotionVectors}, .texture = true},
            {.fields = {GBufferField::Normal, GBufferField::Roughness}, .texture = true},
            {.fields = {GBufferField::DiffuseAlbedo}, .texture = true},
            {.fields = {GBufferField::SpecularAlbedo}, .texture = true},
        });
        break;
    }

    return {
        {"scene", con_scene},
        {"gbuffer", con_gbuffer, ConnectorAccess::compute_read},
        {"src", con_src, ConnectorAccess::compute_read | ConnectorAccess::transfer_src},
        {.name = "specular_hit_distance",
         .connector = con_specular_hit_distance,
         .access = ConnectorAccess::compute_read,
         .optional = true},
    };
}

std::vector<OutputConnectorDescriptor> DLSSNode::describe_outputs(const NodeIOLayout& io_layout) {
    render_extent = io_layout[con_gbuffer]->get_create_info().extent;
    const vk::Extent3D automatic = camera_extent.width != 0 ? camera_extent : render_extent;
    target_extent = vk::Extent3D{out_width != 0 ? out_width : automatic.width,
                                 out_height != 0 ? out_height : automatic.height, 1};
    quality = quality_for_scale(static_cast<float>(target_extent.width) /
                                static_cast<float>(render_extent.width));

    io_layout.register_event_listener(reset_event_pattern,
                                      [this](const GraphEvent::Info&, const GraphEvent::Data&) {
                                          reset = true;
                                          return true;
                                      });

    con_out = ManagedVkImageOut::create(out_format, target_extent);
    return {{"out", con_out, ConnectorAccess::compute_write | ConnectorAccess::transfer_dst}};
}

DLSSNode::NodeStatusFlags DLSSNode::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                                 [[maybe_unused]] const NodeIO& io,
                                                 [[maybe_unused]] const NodeConnectionInfo& info,
                                                 Submission& submission) {
    // NGX refuses a second feature for the same slot while the old one is alive.
    dlss.reset();
    if (mode == Mode::Bypass) {
        return {};
    }

    const std::shared_ptr<ExtensionDLSS>& dlss_extension = get_extension();
    if (!dlss_extension) {
        throw graph_errors::node_error{
            fmt::format("{} is not supported", ExtensionDLSSRayReconstruction::name)};
    }
    if (const std::string& unsupported = dlss_extension->get_unsupported_reason();
        !unsupported.empty()) {
        throw graph_errors::node_error{unsupported};
    }

    const vk::Extent2D target{target_extent.width, target_extent.height};
    if (const auto resolution = dlss_extension->query_resolution(target, quality);
        resolution && (render_extent.width < resolution->min.width ||
                       render_extent.width > resolution->max.width ||
                       render_extent.height < resolution->min.height ||
                       render_extent.height > resolution->max.height)) {
        SPDLOG_WARN("DLSS renders {}x{} for a {}x{} output, outside the {}x{} to {}x{} the mode "
                    "accepts; {}x{} is the intended one",
                    render_extent.width, render_extent.height, target.width, target.height,
                    resolution->min.width, resolution->min.height, resolution->max.width,
                    resolution->max.height, resolution->optimal.width, resolution->optimal.height);
    }

    const DLSSCreateInfo create_info{
        .render_extent = vk::Extent2D{render_extent.width, render_extent.height},
        .target_extent = target,
        .quality = quality,
    };
    try {
        dlss = dlss_extension->create(create_info, submission.get_cmd());
    } catch (const MerianException& e) {
        throw graph_errors::node_error{e.what()};
    }
    reset = true;

    return {};
}

[[nodiscard]] DLSSNode::NodeStatusFlags
DLSSNode::pre_process(const NodeIO& io, [[maybe_unused]] const NodeProcessInfo& info) {
    const SceneHandle& scene = io[con_scene];
    if (!scene || !scene->is_ready()) {
        return {};
    }
    // the camera's resolution is known once the scene is ready
    camera_extent = scene->get_resolution().value_or(vk::Extent3D{});
    if (const vk::Extent3D automatic = camera_extent.width != 0 ? camera_extent : render_extent;
        (out_width == 0 && target_extent.width != automatic.width) ||
        (out_height == 0 && target_extent.height != automatic.height)) {
        return NodeStatusFlagBits::NEEDS_RECONNECT;
    }
    const CameraHandle camera = scene->get_active_camera();
    if (!camera) {
        return {};
    }
    if (camera->get_jitter_sequence() != jitter_sequence) {
        camera->set_jitter_sequence(jitter_sequence);
    }
    const float scale =
        static_cast<float>(target_extent.width) / static_cast<float>(render_extent.width);
    const uint32_t phases = jitter_phases != 0 ? jitter_phases : recommended_jitter_phases(scale);
    if (camera->get_jitter_phases() != phases) {
        camera->set_jitter_phases(phases);
    }

    return {};
}

[[nodiscard]] DLSSNode::NodeStatusFlags DLSSNode::process(
    const NodeIO& io, [[maybe_unused]] const NodeProcessInfo& info, Submission& submission) {
    if (mode == Mode::Bypass) {
        const ImageHandle& src = io[con_src].get_texture()->get_image();
        const ImageHandle& out = io[con_out].get_texture()->get_image();
        cmd_blit_stretch(submission.get_cmd(), src, vk::ImageLayout::eGeneral, src->get_extent(),
                         out, vk::ImageLayout::eGeneral, out->get_extent());
        return {};
    }

    const SceneHandle& scene = io[con_scene];
    if (!scene || !scene->is_ready()) {
        return {};
    }
    const CameraHandle camera = scene->get_active_camera();
    if (!camera) {
        return {};
    }

    const auto gbuffer = io[con_gbuffer];
    DLSSEvalInfo eval_info{
        .color = io[con_src].get_texture()->get_view(),
        .depth = gbuffer->get_view(mode == Mode::RayReconstruction ? GBufferField::ViewDepth
                                                                   : GBufferField::ProjectedDepth),
        .motion_vectors = gbuffer->get_view(GBufferField::MotionVectors),
        .output = io[con_out].get_texture()->get_view(),
        .jitter = camera->get_jitter(),
        .reset = reset,
    };
    if (mode == Mode::RayReconstruction) {
        eval_info.diffuse_albedo = gbuffer->get_view(GBufferField::DiffuseAlbedo);
        eval_info.specular_albedo = gbuffer->get_view(GBufferField::SpecularAlbedo);
        eval_info.normal_roughness = gbuffer->get_view(GBufferField::Normal);
        if (io.is_connected(con_specular_hit_distance)) {
            eval_info.specular_hit_distance =
                io[con_specular_hit_distance].get_texture()->get_view();
        }
        eval_info.world_to_view = camera->get_view_matrix();
        eval_info.view_to_clip = camera->get_projection_matrix();
    }

    try {
        dlss->evaluate(submission.get_cmd(), eval_info);
    } catch (const MerianException& e) {
        throw graph_errors::node_error{e.what()};
    }
    reset = false;

    return {};
}

const std::shared_ptr<ExtensionDLSS>& DLSSNode::get_extension() const {
    static const std::shared_ptr<ExtensionDLSS> BYPASSED;
    switch (mode) {
    case Mode::SuperResolution:
        return dlss_super_resolution;
    case Mode::RayReconstruction:
        return dlss_ray_reconstruction;
    case Mode::Bypass:
        break;
    }
    return BYPASSED;
}

DLSSNode::NodeStatusFlags DLSSNode::properties(Properties& config) {
    bool needs_reconnect = false;

    needs_reconnect |= config.config_uint("width", &out_width, AUTO_RESOLUTION_HINT);
    needs_reconnect |= config.config_uint("height", &out_height, AUTO_RESOLUTION_HINT);
    int mode_index = static_cast<int>(mode);
    if (config.config_options("mode", mode_index, MODE_NAMES, Properties::OptionsStyle::COMBO,
                              "super resolution antialiases and upscales; ray reconstruction also "
                              "denoises and takes the noisy input; bypass stretches the input")) {
        mode = static_cast<Mode>(mode_index);
        needs_reconnect = true;
    }
    needs_reconnect |=
        config.config_enum("output format", out_format, Properties::OptionsStyle::COMBO);

    config.st_separate();
    const auto selected = std::ranges::find(JITTER_SEQUENCES, jitter_sequence);
    int sequence_index = static_cast<int>(
        std::distance(JITTER_SEQUENCES.begin(),
                      selected == JITTER_SEQUENCES.end() ? JITTER_SEQUENCES.begin() : selected));
    if (config.config_options("camera jitter", sequence_index, JITTER_SEQUENCE_NAMES,
                              Properties::OptionsStyle::COMBO,
                              "applied to the active camera; DLSS reconstructs from it")) {
        jitter_sequence = JITTER_SEQUENCES[sequence_index];
    }
    config.config_uint("jitter phases", &jitter_phases,
                       "0 takes the count DLSS asks for at this scale");
    needs_reconnect |= config.config_text("reset event pattern", reset_event_pattern, false,
                                          "drops the temporal history, for cuts and teleports");

    config.st_separate();
    config.output_text("render: {}x{} -> output: {}x{} ({})", render_extent.width,
                       render_extent.height, target_extent.width, target_extent.height,
                       DLSS_QUALITY_NAMES[static_cast<uint32_t>(quality)]);
    if (const std::shared_ptr<ExtensionDLSS>& dlss_extension = get_extension()) {
        if (const std::string& unsupported = dlss_extension->get_unsupported_reason();
            !unsupported.empty()) {
            config.output_text("unsupported: {}", unsupported);
        }
        config.output_text("NGX allocated: {:.1f} MB",
                           static_cast<double>(dlss_extension->get_allocated_memory()) / (1 << 20));
    }

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
