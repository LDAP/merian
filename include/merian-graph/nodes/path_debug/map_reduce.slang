import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.colors.colorspaces;
import merian_shaders.colors.colormaps;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"
#include "merian-graph/nodes/path_debug/map_display.slangh"
#include "merian-shaders/utils/math-constants.slangh"

struct SetReduction {
    float chi2;
    uint dof;
    float lag_product;
    float lag_square;
}

// a tested set's bin at a batch end: its residual, and with the previous one where the bin lies
// inside the set's support
SetReduction record_batch(const uint set,
                          const uint bin,
                          const uint2 cell,
                          const float solid_angle) {
    SetReduction r = { 0.f, 0, 0.f, 0.f };
    const float sum_recip =
        asfloat(maps[path_debug_map_set_at(set, PATH_DEBUG_SET_REGION_RECIP, bin)]);
    const float batch_draws = float(stats[path_debug_set_stat(set, PATH_DEBUG_SET_DRAWS)] -
                                    stats[path_debug_set_stat(set, PATH_DEBUG_SET_BATCH_DRAWS)]);
    const uint start_at = path_debug_map_set_at(set, PATH_DEBUG_SET_REGION_RECIP_START, bin);
    const uint residual2_at = path_debug_map_set_at(set, PATH_DEBUG_SET_REGION_RESIDUAL2, bin);
    const uint prev_at = path_debug_map_set_at(set, PATH_DEBUG_SET_REGION_RESIDUAL_PREV, bin);
    const float residual = (sum_recip - asfloat(maps[start_at])) - batch_draws * solid_angle;
    if (stats[PATH_DEBUG_STATS_BATCHES] > 0 &&
        path_debug_support(set, cell) == PATH_DEBUG_SUPPORT_INSIDE) {
        const float prev = asfloat(maps[prev_at]);
        r.lag_product = residual * prev;
        r.lag_square = prev * prev;
    }
    maps[residual2_at] = asuint(asfloat(maps[residual2_at]) + residual * residual);
    maps[prev_at] = asuint(residual);
    maps[start_at] = asuint(sum_recip);
    return r;
}

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint bin = gid.x;
    const bool in_map = bin < params.map_res * params.map_res;
    const uint2 cell = path_debug_map_cell_of(bin);
    const float solid_angle =
        in_map ? dir_map_bin_solid_angle(params.map_transform, cell, params.map_res) : 0.f;

    float shared = 0.f;
    float nee_max = 0.f;
    float moment = 0.f;
    float radiance = 0.f;
    float pdf_part = 0.f;
    float eval_part = 0.f;
    float tv_check = 0.f;
    float tv_integrand = 0.f;
    float check_chi2 = 0.f;
    uint check_dof = 0;
    SetReduction sets[PATH_DEBUG_TESTED_SETS];
    const bool batch_end = params.map_batch_end != 0;
    const SetReduction untested = { 0.f, 0, 0.f, 0.f };
    for (uint set = 0; set < PATH_DEBUG_TESTED_SETS; set++) {
        sets[set] = in_map && batch_end ? record_batch(set, bin, cell, solid_angle) : untested;
    }
    if (in_map) {
        const PathDebugBinTest scatter_test =
            path_debug_recorded_test(PATH_DEBUG_SET_SCATTER, cell, solid_angle, batch_end);
        const PathDebugBinTest nee_test = path_debug_nee_test(cell, solid_angle, batch_end);
        if (scatter_test.state == PATH_DEBUG_BIN_TESTED) {
            sets[PATH_DEBUG_SET_SCATTER].chi2 = scatter_test.z * scatter_test.z;
            sets[PATH_DEBUG_SET_SCATTER].dof = 1;
        }
        if (nee_test.state == PATH_DEBUG_BIN_TESTED) {
            sets[PATH_DEBUG_SET_NEE].chi2 = nee_test.z * nee_test.z;
            sets[PATH_DEBUG_SET_NEE].dof = 1;
        }
        if (params.sphere_channel != PATH_DEBUG_MAP_NONE && params.sphere_scale != 0) {
            maps[path_debug_map_sphere_at(bin)] =
                asuint(path_debug_map_value(params.sphere_channel, cell, solid_angle));
        }

        const float sampled =
            path_debug_map_value(PATH_DEBUG_MAP_SAMPLED_SCATTER, cell, solid_angle);
        const float pdf = path_debug_map_value(PATH_DEBUG_MAP_PDF, cell, solid_angle);
        const float check = path_debug_map_value(PATH_DEBUG_MAP_CHECK, cell, solid_angle);
        const float eval_density = path_debug_map_value(PATH_DEBUG_MAP_EVAL, cell, solid_angle);
        const float target =
            path_debug_map_value(PATH_DEBUG_MAP_INTEGRAND_NORMALIZED, cell, solid_angle);
        shared = max(max(max(sampled, pdf), max(check, eval_density)),
                     max(max(target, path_debug_map_value(PATH_DEBUG_MAP_SAMPLED_GUIDING, cell,
                                                          solid_angle)),
                         path_debug_map_value(PATH_DEBUG_MAP_RECORDED_SCATTER, cell,
                                              solid_angle)));
        nee_max = max(path_debug_map_value(PATH_DEBUG_MAP_SAMPLED_NEE, cell, solid_angle),
                  path_debug_map_value(PATH_DEBUG_MAP_RECORDED_NEE, cell, solid_angle));
        moment = path_debug_map_value(PATH_DEBUG_MAP_SECOND_MOMENT, cell, solid_angle);
        radiance = path_debug_map_value(PATH_DEBUG_MAP_RADIANCE, cell, solid_angle);
        pdf_part = pdf * solid_angle;
        eval_part = yuv_luminance(path_debug_map_eval(bin)) * solid_angle;
        // the pdf's normalization is known from the second frame after a reset
        const float pdf_norm = asfloat(stats[PATH_DEBUG_STATS_PDF_NORM]);
        if (pdf_norm > 0.f) {
            const float pdf_mass = pdf_part / pdf_norm;
            if (stats[PATH_DEBUG_STATS_CHECK_SAMPLES] > 0) {
                tv_check = abs(pdf_mass - check * solid_angle);
            }
        }
        tv_integrand = abs(sampled - target) * solid_angle;

        const PathDebugBinTest test = path_debug_check_test(cell, solid_angle);
        if (test.state == PATH_DEBUG_BIN_TESTED) {
            check_chi2 = test.z * test.z;
            check_dof = 1;
        }
    }

    const float wave_eval_part = WaveActiveSum(eval_part);
    const float wave_tv_check = WaveActiveSum(tv_check);
    const float wave_tv_integrand = WaveActiveSum(tv_integrand);
    const float wave_shared = WaveActiveMax(shared);
    const float wave_nee_max = WaveActiveMax(nee_max);
    const float wave_moment = WaveActiveMax(moment);
    const float wave_radiance = WaveActiveMax(radiance);
    const float wave_integral = WaveActiveSum(pdf_part);
    const float wave_check_chi2 = WaveActiveSum(check_chi2);
    const uint wave_check_dof = WaveActiveSum(check_dof);
    if (WaveIsFirstLane()) {
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_SHARED, wave_shared);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_NEE, wave_nee_max);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_MOMENT, wave_moment);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_RADIANCE, wave_radiance);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_EVAL_INTEGRAL, wave_eval_part);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_TV_CHECK, wave_tv_check);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_TV_INTEGRAND,
                                    wave_tv_integrand);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_PDF_INTEGRAL, wave_integral);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_CHECK_CHI2, wave_check_chi2);
        InterlockedAdd(stats[PATH_DEBUG_STATS_CHECK_DOF], wave_check_dof);
    }
    for (uint set = 0; set < PATH_DEBUG_TESTED_SETS; set++) {
        const float wave_chi2 = WaveActiveSum(sets[set].chi2);
        const uint wave_dof = WaveActiveSum(sets[set].dof);
        const float wave_lag_product = WaveActiveSum(sets[set].lag_product);
        const float wave_lag_square = WaveActiveSum(sets[set].lag_square);
        // the NEE test reports for both of its sets
        const uint reported = set == PATH_DEBUG_SET_NEE_ENV ? PATH_DEBUG_SET_NEE : set;
        if (WaveIsFirstLane()) {
            path_debug_atomic_add_float(stats, stats_f,
                                        path_debug_set_stat(reported, PATH_DEBUG_SET_CHI2),
                                        wave_chi2);
            InterlockedAdd(stats[path_debug_set_stat(reported, PATH_DEBUG_SET_DOF)], wave_dof);
            path_debug_atomic_add_float(stats, stats_f,
                                        path_debug_set_stat(reported, PATH_DEBUG_SET_LAG_PRODUCT),
                                        wave_lag_product);
            path_debug_atomic_add_float(stats, stats_f,
                                        path_debug_set_stat(reported, PATH_DEBUG_SET_LAG_SQUARE),
                                        wave_lag_square);
        }
    }
}
