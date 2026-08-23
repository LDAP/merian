#include "merian-graph/nodes/guiding/mcpg/mcpg_guiding_model.hpp"

#include <fmt/format.h>

namespace merian {

namespace {

constexpr const char* GUIDING_MODULE = "merian-graph/nodes/guiding/mcpg/mcpg-guiding.slang";

} // namespace

void MCPGGuidingModel::initialize(const ContextHandle& context,
                                  const ResourceAllocatorHandle& allocator) {
    this->compile_context = context->get_shader_compile_context();
    this->allocator = allocator;
    // splitting keys from the payload costs a second indirection that only pays off off AMD
    mc_split_storage = lc_split_storage = !context->get_device()->get_physical_device()->is_amd();
    recreate_grids();
}

SlangCompositionHandle MCPGGuidingModel::query_device_support_composition() {
    const auto composition = SlangComposition::create();
    composition->add_composition(MCPG::query_device_support_composition());
    composition->add_composition(HashedIrradianceCache::query_device_support_composition());
    composition->add_module_from_path(GUIDING_MODULE);
    return composition;
}

void MCPGGuidingModel::recreate_grids() {
    mcpg = std::make_shared<MCPG>(compile_context, allocator, mc_buffer_size, mc_split_storage);
    irr_cache = std::make_shared<HashedIrradianceCache>(compile_context, allocator, lc_buffer_size,
                                                        lc_probe_count, lc_stochastic_interpolation,
                                                        lc_split_storage);
}

SlangCompositionHandle MCPGGuidingModel::get_composition() const {
    const auto composition = SlangComposition::create();
    composition->add_composition(mcpg->get_composition());
    composition->add_composition(irr_cache->get_composition());
    composition->add_module_from_path(GUIDING_MODULE);
    composition->add_module_from_string(
        "mcpg_guiding_constants",
        fmt::format("namespace merian {{\n"
                    "export static const int merian_guiding_mc_samples = {};\n"
                    "export static const float merian_guiding_probability = {};\n"
                    "export static const bool merian_guiding_scale_with_alpha = {};\n"
                    "export static const float merian_guiding_alpha_threshold = {};\n"
                    "export static const bool merian_guiding_missing_light_heuristic = {};\n"
                    "export static const bool merian_guiding_light_cache_tail = {};\n"
                    "export static const float merian_guiding_lc_min_pdf = {};\n"
                    "}}\n"
                    "export static const float dir_guide_prior = {};",
                    mc_samples, probability, scale_with_alpha ? "true" : "false", alpha_threshold,
                    missing_light_heuristic ? "true" : "false", light_cache_tail ? "true" : "false",
                    lc_min_pdf, dir_guide_prior));
    return composition;
}

std::vector<std::string> MCPGGuidingModel::get_slang_imports() const {
    return {GUIDING_MODULE};
}

std::string MCPGGuidingModel::get_type_name() const {
    return fmt::format("merian::MCPGGuiding<{}u, {}u, {}, {}u, {}u, {}u, {}, {}, {}u>",
                       mcpg->get_buffer_size(), mc_probe_count, mc_split_storage ? "true" : "false",
                       mc_locality_bits, irr_cache->get_buffer_size(), lc_probe_count,
                       lc_stochastic_interpolation ? "true" : "false",
                       lc_split_storage ? "true" : "false", lc_locality_bits);
}

void MCPGGuidingModel::write_to(ShaderCursor cursor) {
    mcpg->write_to(cursor["mcpg"]);
    irr_cache->write_to(cursor["irr_cache"]);
}

void MCPGGuidingModel::reset(const CommandBufferHandle& cmd) {
    mcpg->reset(cmd);
    irr_cache->reset(cmd);
}

bool MCPGGuidingModel::properties(Properties& props) {
    bool constants_changed = false;
    bool recreate = false;

    if (props.st_begin_child("mc", "Markov Chain Path Guiding",
                             Properties::ChildFlagBits::DEFAULT_OPEN)) {
        constants_changed |= props.config_percent("ML prior", dir_guide_prior);
        constants_changed |= props.config_int("MC samples", mc_samples, "", 0, 30);
        constants_changed |= props.config_float(
            "guiding probability", probability,
            "Probability of drawing the scatter direction from the guiding lobes.", 0.01f, 0.f,
            1.f);
        constants_changed |=
            props.config_bool("scale with alpha", scale_with_alpha,
                              "Scale the guiding probability with the lobe width.");
        constants_changed |=
            props.config_float("alpha threshold", alpha_threshold,
                               "Do not guide below this lobe width.", 0.01f, 0.f, 1.f);
        constants_changed |= props.config_bool(
            "missing light heuristic", missing_light_heuristic,
            "Flood the Markov chains with invalidated states when no light is detected.");
        recreate |= props.config_uint("adaptive grid buf size", mc_buffer_size,
                                      "Buffer size backing the hash grid.");
        constants_changed |=
            props.config_uint("MC probe count", mc_probe_count,
                              "Slots probed before evicting (open addressing).", 1u, 32u);
        recreate |= props.config_bool("split keys/payload", mc_split_storage,
                                      "Store hash+stamp separately from the payload "
                                      "(probe-friendly) instead of one combined record per slot.");
        constants_changed |= props.config_uint(
            "locality bits", mc_locality_bits,
            "Give each 2^n-wide cell tile a contiguous Morton-ordered slot range so nearby "
            "cells share cache lines (0 = scatter every cell).",
            0u, 5u);
        if (mcpg) {
            mcpg->properties(props);
        }
        props.st_end_child();
    }

    if (props.st_begin_child("lc", "Light cache", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        constants_changed |= props.config_bool("surf: use LC", light_cache_tail,
                                               "Use the light cache for the path tail.");
        recreate |=
            props.config_uint("LC buffer size", lc_buffer_size,
                              "Number of cache slots backing the hash grid.", 1u, 100000000u);
        recreate |= props.config_uint("LC probe count", lc_probe_count,
                                      "Slots probed before evicting (open addressing).", 1u, 16u);
        recreate |= props.config_bool("LC stochastic interpolation", lc_stochastic_interpolation,
                                      "Jitter the grid cell per sample (smoother but noisier) "
                                      "instead of snapping to the nearest cell.");
        recreate |= props.config_bool("LC split keys/payload", lc_split_storage,
                                      "Store hash+stamp separately from the payload "
                                      "(probe-friendly) instead of one combined record per slot.");
        constants_changed |= props.config_uint(
            "LC locality bits", lc_locality_bits,
            "Give each 2^n-wide cell tile a contiguous Morton-ordered slot range so nearby "
            "cells share cache lines (0 = scatter every cell).",
            0u, 5u);
        constants_changed |= props.config_float(
            "LC min pdf", lc_min_pdf,
            "Increase to reduce fireflies in the irradiance cache and bias the guiding towards "
            "direct light, especially useful for short maximum path lengths.",
            0.1f, 0.0f);
        if (irr_cache) {
            irr_cache->properties(props);
        }
        props.st_end_child();
    }

    if (recreate && mcpg) {
        recreate_grids();
    }

    return constants_changed || recreate;
}

} // namespace merian
