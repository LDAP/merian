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
                                                        lc_split_storage, lc_locality_bits);
}

SlangCompositionHandle MCPGGuidingModel::get_composition() const {
    const auto composition = SlangComposition::create();
    composition->add_composition(mcpg->get_composition());
    composition->add_composition(irr_cache->get_composition());
    composition->add_module_from_path(GUIDING_MODULE);
    composition->add_module_from_string(
        "mcpg_guiding_constants",
        fmt::format("import {};\n"
                    "namespace merian {{\n"
                    "export static const int merian_guiding_mc_samples = {};\n"
                    "export static const float merian_guiding_weight_exponent = {};\n"
                    "export static const bool merian_guiding_missing_light_heuristic = {};\n"
                    "export static const MCPGReplacement merian_guiding_replacement = "
                    "MCPGReplacement({});\n"
                    "export static const float merian_guiding_lc_min_pdf = {};\n"
                    "}}\n"
                    "export static const float dir_guide_prior = {};\n"
                    "export static const float mc_conf_z = {};\n"
                    "export static const bool mc_welford_chord = {};",
                    slang_import_spelling(GUIDING_MODULE), mc_samples, weight_exponent,
                    missing_light_heuristic ? "true" : "false", replacement, lc_min_pdf,
                    dir_guide_prior, mc_conf_z, mc_welford_chord ? "true" : "false"));
    return composition;
}

std::vector<std::string> MCPGGuidingModel::get_slang_imports() const {
    return {slang_import_spelling(GUIDING_MODULE)};
}

std::string MCPGGuidingModel::get_type_name() const {
    return fmt::format("merian::MCPGGuiding<{}u, {}u, {}, {}u, {}u, {}u, {}, {}, {}u>",
                       mcpg->get_buffer_size(), mc_probe_count, mc_split_storage ? "true" : "false",
                       mc_locality_bits, irr_cache->get_buffer_size(), lc_probe_count,
                       lc_stochastic_interpolation ? "true" : "false",
                       lc_split_storage ? "true" : "false", lc_locality_bits);
}

void MCPGGuidingModel::write_to(ShaderCursor cursor) {
    mcpg->set_grid_params(grid_params);
    irr_cache->set_grid_params(lc_params);
    mcpg->write_to(cursor["mcpg"]);
    irr_cache->write_to(cursor["irr_cache"]);
    cursor["debug_selector"] = debug_view;
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
        constants_changed |= props.config_float(
            "ML prior", dir_guide_prior,
            "Prior on a chain's lobe width, in squared distance to its target.", 0.01f, 0.f);
        constants_changed |= props.config_int("MC samples", mc_samples, "", 1, 30);
        constants_changed |= props.config_float(
            "width confidence z", mc_conf_z,
            "Sample the lobe width's upper confidence limit at this standard normal quantile "
            "instead of the maximum likelihood width, so a chain with little behind it proposes a "
            "wide lobe rather than a confident one. 0 disables, 1.6449 is the 95 % limit.",
            0.01f, 0.f, 4.f);
        constants_changed |= props.config_bool(
            "welford chord", mc_welford_chord,
            "Measure a new sample's chord against the mean direction before and after it is folded "
            "in; on its own the updated mean reports too small a spread.");
        constants_changed |= props.config_float(
            "weight exponent", weight_exponent,
            "Exponent on a chain's weight where it selects among candidates.", 0.1f, 0.25f, 4.f);
        constants_changed |= props.config_bool(
            "missing light heuristic", missing_light_heuristic,
            "Invalidate the selected chain's cell where the light it aims at went missing.");
        constants_changed |= props.config_options(
            "replacement", replacement,
            {"always (Alber et al. 2025)", "mutated state / replaced state"},
            Properties::OptionsStyle::COMBO,
            "A mutated state overwrites the state in the slot it is stored to either always or "
            "with probability its weight over that state's weight, so a dim state cannot displace "
            "a brighter one.");
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
        grid_params.properties(props);
        props.st_end_child();
    }

    if (props.st_begin_child("lc", "Light cache", Properties::ChildFlagBits::DEFAULT_OPEN)) {
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
        if (props.config_uint("LC locality bits", lc_locality_bits,
                              "Give each 2^n-wide cell tile a contiguous Morton-ordered slot "
                              "range so nearby cells share cache lines (0 = scatter every cell).",
                              0u, 5u)) {
            if (irr_cache) {
                irr_cache->set_locality_bits(lc_locality_bits);
            }
            constants_changed = true;
        }
        constants_changed |= props.config_float(
            "LC min pdf", lc_min_pdf,
            "Increase to reduce fireflies in the irradiance cache and bias the guiding towards "
            "direct light, especially useful for short maximum path lengths.",
            0.1f, 0.0f);
        lc_params.properties(props);
        props.st_end_child();
    }

    props.st_separate("Debug");
    props.config_options(
        "debug output", debug_view,
        {"light cache", "mc grid", "mc lod", "mc weight", "mc mean direction", "mc cos", "mc N",
         "mc mv", "lc normal bin (actual)", "lc normal bin (selected)", "unresolved"},
        Properties::OptionsStyle::COMBO, "What the renderer shows as its guiding debug output.");

    if (recreate && mcpg) {
        recreate_grids();
    }

    return constants_changed || recreate;
}

} // namespace merian
