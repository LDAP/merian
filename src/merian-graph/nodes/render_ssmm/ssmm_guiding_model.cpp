#include "merian-graph/nodes/render_ssmm/ssmm_guiding_model.hpp"

#include "merian-graph/nodes/render_ssmm/ssmm_guiding.slangh"
#include "merian/utils/properties.hpp"

#include <fmt/format.h>

namespace merian {

namespace {

constexpr const char* GUIDING_MODULE = "merian-graph/nodes/render_ssmm/ssmm-guiding.slang";

} // namespace

void SSMMGuidingModel::initialize([[maybe_unused]] const ContextHandle& context,
                                  const ResourceAllocatorHandle& allocator) {
    this->allocator = allocator;
}

SlangCompositionHandle SSMMGuidingModel::query_device_support_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(GUIDING_MODULE);
    return composition;
}

SlangCompositionHandle SSMMGuidingModel::get_composition() const {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(GUIDING_MODULE);
    composition->add_module_from_string(
        "ssmm_guiding_constants",
        fmt::format("namespace merian {{\n"
                    "export static const int merian_ssmm_group_size = {};\n"
                    "export static const float merian_ssmm_reuse_radius = {};\n"
                    "export static const float merian_ssmm_probability = {};\n"
                    "export static const float merian_ssmm_alpha_threshold = {};\n"
                    "export static const uint merian_ssmm_max_n = {}u;\n"
                    "export static const float merian_ssmm_min_alpha = {};\n"
                    "export static const float merian_ssmm_prior_n = {};\n"
                    "}}",
                    group_size, reuse_radius, probability, alpha_threshold, max_n, min_alpha,
                    prior_n));
    return composition;
}

std::vector<std::string> SSMMGuidingModel::get_slang_imports() const {
    return {GUIDING_MODULE};
}

std::string SSMMGuidingModel::get_type_name() const {
    return "merian::SSMMGuiding";
}

void SSMMGuidingModel::on_extent(const vk::Extent3D& new_extent) {
    if (new_extent == extent) {
        return;
    }
    extent = new_extent;

    const vk::DeviceSize size = vk::DeviceSize(extent.width) * extent.height * sizeof(SSMCState);
    const auto usage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    for (uint32_t i = 0; i < states.size(); i++) {
        states[i] = allocator->create_buffer(size, usage, MemoryMappingType::NONE,
                                             fmt::format("SSMMGuidingModel::states[{}]", i));
    }
}

void SSMMGuidingModel::set_gbuffer(const ShaderObjectHandle& gbuffer) {
    this->gbuffer = gbuffer;
}

void SSMMGuidingModel::write_to(ShaderCursor cursor) {
    cursor["gbuffer"] = gbuffer;
    cursor["prev"] = states[(frame + 1) & 1u];
    cursor["next"] = states[frame & 1u];
    frame++;
}

void SSMMGuidingModel::reset(const CommandBufferHandle& cmd) {
    std::array<vk::BufferMemoryBarrier2, 2> barriers;
    for (uint32_t i = 0; i < states.size(); i++) {
        cmd->fill(states[i]);
        barriers[i] = states[i]->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    }
    cmd->barrier(barriers);
    frame = 0;
}

std::array<vk::BufferMemoryBarrier2, 2> SSMMGuidingModel::carry_barriers() const {
    std::array<vk::BufferMemoryBarrier2, 2> barriers;
    for (uint32_t i = 0; i < states.size(); i++) {
        barriers[i] = states[i]->buffer_barrier2(
            vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eMemoryWrite,
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
    }
    return barriers;
}

bool SSMMGuidingModel::properties(Properties& props) {
    bool constants_changed = false;

    constants_changed |= props.config_int("group size", group_size,
                                          "Neighbour fits resampled towards each vertex.", 1, 16);
    constants_changed |= props.config_float(
        "reuse radius", reuse_radius,
        "Standard deviation of the neighbourhood the resampling reaches into, in pixels.", 1.f,
        64.f);
    constants_changed |=
        props.config_percent("probability", probability,
                             "Probability the integrator gives the fitted lobe over the BSDF.");
    constants_changed |= props.config_float(
        "alpha threshold", alpha_threshold,
        "No guiding below this GGX alpha: the BSDF is more peaked than any fitted lobe.", 0.f, 1.f);
    constants_changed |=
        props.config_uint("max N", max_n, "Samples folded into the exponentially weighted fit.");
    constants_changed |= props.config_float("min alpha", min_alpha,
                                            "Floor on the exponential weight, so the fit keeps "
                                            "adapting once max N is reached.",
                                            0.f, 1.f);
    constants_changed |= props.config_percent("ML prior", prior_n);

    return constants_changed;
}

} // namespace merian
