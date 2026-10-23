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

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint bin = gid.x;
    const bool in_map = bin < params.map_res * params.map_res;
    const uint2 cell = path_debug_map_cell_of(bin);
    const float solid_angle =
        in_map ? dir_map_bin_solid_angle(params.map_transform, cell, params.map_res) : 0.f;

    float shared = 0.f;
    float integrand = 0.f;
    float moment = 0.f;
    float radiance = 0.f;
    float mean = 0.f;
    float pdf_part = 0.f;
    float eval_part = 0.f;
    float tv_eval = 0.f;
    float tv_check = 0.f;
    float tv_records = 0.f;
    float chi2 = 0.f;
    uint dof = 0;
    float lag_product = 0.f;
    float lag_square = 0.f;
    if (in_map) {
        const float density = path_debug_map_value(PATH_DEBUG_MAP_DENSITY, cell, solid_angle);
        const float pdf = path_debug_map_value(PATH_DEBUG_MAP_PDF, cell, solid_angle);
        const float check = path_debug_map_value(PATH_DEBUG_MAP_CHECK, cell, solid_angle);
        const float eval_density = path_debug_map_value(PATH_DEBUG_MAP_EVAL, cell, solid_angle);
        shared = max(max(max(density, pdf), max(check, eval_density)),
                     max(path_debug_map_value(PATH_DEBUG_MAP_INTEGRAND_NORMALIZED, cell,
                                              solid_angle),
                         path_debug_map_value(PATH_DEBUG_MAP_RECORDED_PDF, cell, solid_angle)));
        integrand = path_debug_map_value(PATH_DEBUG_MAP_INTEGRAND, cell, solid_angle);
        moment = path_debug_map_value(PATH_DEBUG_MAP_SECOND_MOMENT, cell, solid_angle);
        radiance = path_debug_map_value(PATH_DEBUG_MAP_RADIANCE_COLOR, cell, solid_angle);
        mean = path_debug_map_value(PATH_DEBUG_MAP_MEAN, cell, solid_angle);
        pdf_part = pdf * solid_angle;
        eval_part = yuv_luminance(path_debug_map_eval(bin)) * solid_angle;
        // the pdf's normalization is known from the second frame after a reset
        const float pdf_norm = asfloat(stats[PATH_DEBUG_STATS_PDF_NORM]);
        const float pdf_mass = pdf_norm > 0.f ? pdf_part / pdf_norm : 0.f;
        if (solid_angle > 0.f && pdf_norm > 0.f) {
            tv_eval = abs(pdf_mass - eval_density * solid_angle);
            if (stats[PATH_DEBUG_STATS_CHECK_SAMPLES] > 0) {
                tv_check = abs(pdf_mass - check * solid_angle);
            }
            if (path_debug_bsdf_comparable()) {
                tv_records = abs(pdf_mass - density * solid_angle);
            }
        }

        const uint count_raw = maps[path_debug_map_count_at(bin)];
        const float count = float(count_raw);
        const float sum_recip = asfloat(maps[path_debug_map_recip_at(bin)]);
        const bool batch_end = params.map_batch_end != 0;
        if (batch_end) {
            const float batch_draws =
                float(stats[PATH_DEBUG_STATS_DRAWS] - stats[PATH_DEBUG_STATS_BATCH_DRAWS]);
            const float batch_samples =
                float(stats[PATH_DEBUG_STATS_SAMPLES] - stats[PATH_DEBUG_STATS_BATCH_SAMPLES]);
            const float residual =
                params.map_pdf_ref == PATH_DEBUG_PDF_REF_RECORDED
                    ? (sum_recip - asfloat(maps[path_debug_map_recip_start_at(bin)])) -
                          batch_draws * solid_angle
                    : (count - float(maps[path_debug_map_count_start_at(bin)])) -
                          batch_samples * pdf_mass;
            if (stats[PATH_DEBUG_STATS_BATCHES] > 0) {
                const float prev = asfloat(maps[path_debug_map_residual_prev_at(bin)]);
                lag_product = residual * prev;
                lag_square = prev * prev;
            }
            maps[path_debug_map_residual2_at(bin)] =
                asuint(asfloat(maps[path_debug_map_residual2_at(bin)]) + residual * residual);
            maps[path_debug_map_residual_prev_at(bin)] = asuint(residual);
            maps[path_debug_map_count_start_at(bin)] = count_raw;
            maps[path_debug_map_recip_start_at(bin)] = asuint(sum_recip);
        }

        const PathDebugBinTest test = path_debug_bin_test(cell, solid_angle, batch_end);
        if (test.state == PATH_DEBUG_BIN_TESTED) {
            chi2 = test.z * test.z;
            dof = 1;
        } else if (test.state == PATH_DEBUG_BIN_SPIKE) {
            path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_SPIKE_MASS,
                                        (count - test.expected) /
                                            max(float(stats[PATH_DEBUG_STATS_SAMPLES]), 1.f));
            InterlockedAdd(stats[PATH_DEBUG_STATS_SPIKE_BINS], 1);
        }

        // the polar profile
        const float4 center =
            dir_map_dir(params.map_transform, (float2(cell) + 0.5f) / float(params.map_res));
        if (solid_angle > 0.f && center.w != 0.f) {
            const float theta = acos(clamp(center.z, -1.f, 1.f));
            const uint band = min(uint(theta / MERIAN_M_PI * PATH_DEBUG_THETA_BANDS),
                                  PATH_DEBUG_THETA_BANDS - 1);
            const uint base = PATH_DEBUG_STATS_THETA + PATH_DEBUG_THETA_UINTS * band;
            InterlockedAdd(stats[base + 0], count_raw);
            path_debug_atomic_add_float(stats, stats_f, base + 1, pdf_part);
            path_debug_atomic_add_float(stats, stats_f, base + 2, solid_angle);
        }
    }

    const float wave_eval_part = WaveActiveSum(eval_part);
    const float wave_tv_eval = WaveActiveSum(tv_eval);
    const float wave_tv_check = WaveActiveSum(tv_check);
    const float wave_tv_records = WaveActiveSum(tv_records);
    const float wave_shared = WaveActiveMax(shared);
    const float wave_integrand = WaveActiveMax(integrand);
    const float wave_moment = WaveActiveMax(moment);
    const float wave_radiance = WaveActiveMax(radiance);
    const float wave_mean = WaveActiveMax(mean);
    const float wave_integral = WaveActiveSum(pdf_part);
    const float wave_chi2 = WaveActiveSum(chi2);
    const uint wave_dof = WaveActiveSum(dof);
    const float wave_lag_product = WaveActiveSum(lag_product);
    const float wave_lag_square = WaveActiveSum(lag_square);
    if (WaveIsFirstLane()) {
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_SHARED, wave_shared);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_EVAL_INTEGRAL, wave_eval_part);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_TV_EVAL, wave_tv_eval);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_TV_CHECK, wave_tv_check);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_TV_RECORDS, wave_tv_records);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_INTEGRAND, wave_integrand);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_MOMENT, wave_moment);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_RADIANCE, wave_radiance);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_MEAN, wave_mean);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_PDF_INTEGRAL, wave_integral);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_CHI2, wave_chi2);
        InterlockedAdd(stats[PATH_DEBUG_STATS_CHI2_DOF], wave_dof);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_LAG_PRODUCT, wave_lag_product);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_LAG_SQUARE, wave_lag_square);
    }
}
