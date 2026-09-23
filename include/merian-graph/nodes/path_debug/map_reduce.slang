import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.colors.colorspaces;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"
#include "merian-graph/nodes/path_debug/map_test.slangh"
#include "merian-shaders/utils/math-constants.slangh"

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint bin = gid.x;
    const bool in_map = bin < params.map_res * params.map_res;
    const uint2 cell = uint2(bin % params.map_res, bin / params.map_res);
    const float solid_angle =
        in_map ? dir_map_bin_solid_angle(params.map_transform, cell, params.map_res) : 0.f;

    float density = 0.f;
    float contrib = 0.f;
    float mean = 0.f;
    float pdf = 0.f;
    float pdf_part = 0.f;
    float check = 0.f;
    float eval_density = 0.f;
    float eval_part = 0.f;
    float tv_eval = 0.f;
    float tv_check = 0.f;
    float tv_records = 0.f;
    if (in_map) {
        const uint total = max(stats[PATH_DEBUG_STATS_SAMPLES], 1u);
        const uint count = maps[path_debug_map_count_at(bin)];
        const uint c = path_debug_map_contrib_at(bin);
        const float lum = yuv_luminance(
            float3(asfloat(maps[c + 0]), asfloat(maps[c + 1]), asfloat(maps[c + 2])));
        pdf = abs(asfloat(maps[path_debug_map_pdf_at(bin)]));
        const float eval_raw = asfloat(maps[path_debug_map_eval_at(bin)]);
        if (solid_angle > 0.f) {
            density = float(count) / (float(total) * solid_angle);
            contrib = lum / (float(total) * solid_angle);
            pdf_part = pdf * solid_angle;
            check = float(maps[path_debug_map_check_at(bin)]) /
                    (float(max(stats[PATH_DEBUG_STATS_CHECK_SAMPLES], 1u)) * solid_angle);
            eval_part = eval_raw * solid_angle;
            const float eval_norm = asfloat(stats[PATH_DEBUG_STATS_EVAL_NORM]);
            eval_density = eval_norm > 0.f ? eval_raw / eval_norm : 0.f;
            const float pdf_mass =
                pdf * solid_angle / max(asfloat(stats[PATH_DEBUG_STATS_PDF_NORM]), 1e-6f);
            tv_eval = abs(pdf_mass - eval_density * solid_angle);
            if (stats[PATH_DEBUG_STATS_CHECK_SAMPLES] > 0) {
                tv_check = abs(pdf_mass - check * solid_angle);
            }
            if (path_debug_shared_support(true)) {
                tv_records = abs(pdf_mass - float(count) / float(total));
            }
        }
        mean = count > 0 ? lum / float(count) : 0.f;
    }

    float chi2 = 0.f;
    uint dof = 0;
    float lag_product = 0.f;
    float lag_square = 0.f;
    const float omega = solid_angle;
    if (in_map) {
        const bool recorded = params.map_pdf_ref == PATH_DEBUG_PDF_REF_RECORDED;
        const uint count_raw = maps[path_debug_map_count_at(bin)];
        const float count = float(count_raw);
        const float sum_recip = asfloat(maps[path_debug_map_recip_at(bin)]);
        const float norm = max(asfloat(stats[PATH_DEBUG_STATS_PDF_NORM]), 1e-6f);
        const float count_share = pdf * solid_angle / norm;
        const bool batch_end = params.map_batch_end != 0;

        if (batch_end) {
            const float residual =
                recorded ? (sum_recip - asfloat(maps[path_debug_map_recip_start_at(bin)])) -
                               float(stats[PATH_DEBUG_STATS_DRAWS] -
                                     stats[PATH_DEBUG_STATS_BATCH_DRAWS]) *
                                   solid_angle
                         : (count - float(maps[path_debug_map_count_start_at(bin)])) -
                               float(stats[PATH_DEBUG_STATS_SAMPLES] -
                                     stats[PATH_DEBUG_STATS_BATCH_SAMPLES]) *
                                   count_share;
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

        if (recorded) {
            // E[sum 1/pdf] = draws * omega inside the support
            const float expected = float(stats[PATH_DEBUG_STATS_DRAWS]) * solid_angle;
            const float variance = path_debug_residual_variance(
                bin, true, asfloat(maps[path_debug_map_recip2_at(bin)]), batch_end);
            if (path_debug_shared_support(false) && variance > 0.f && count >= 5.f &&
                expected >= 1.f && path_debug_support_interior(cell)) {
                const float z = (sum_recip - expected) / sqrt(variance);
                chi2 = z * z;
                dof = 1;
                InterlockedAdd(stats[PATH_DEBUG_STATS_TESTED], count_raw);
            }
        } else if (!path_debug_maps_hold(PATH_DEBUG_SEEN_CONNECTION |
                                         PATH_DEBUG_SEEN_RESAMPLED)) {
            const bool resolved = asfloat(maps[path_debug_map_pdf_at(bin)]) >= 0.f;
            const bool shared_support = path_debug_shared_support(true);
            const float expected = float(stats[PATH_DEBUG_STATS_SAMPLES]) * count_share;
            const bool spike = shared_support && path_debug_is_spike(count, expected);
            if (spike) {
                path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_SPIKE_MASS,
                                            (count - expected) /
                                                max(float(stats[PATH_DEBUG_STATS_SAMPLES]), 1.f));
                InterlockedAdd(stats[PATH_DEBUG_STATS_SPIKE_BINS], 1);
            }
            const float variance = path_debug_residual_variance(bin, false, expected, batch_end);
            if (shared_support && resolved && !spike && expected >= 5.f && variance > 0.f) {
                const float diff = count - expected;
                chi2 = diff * diff / variance;
                dof = 1;
                InterlockedAdd(stats[PATH_DEBUG_STATS_TESTED], count_raw);
            }
        }
    }

    if (in_map && omega > 0.f) {
        const float2 uv = (float2(cell) + 0.5f) / float(params.map_res);
        const float4 center = dir_map_dir(params.map_transform, uv);
        if (center.w != 0.f) {
            const float theta = acos(clamp(center.z, -1.f, 1.f));
            const uint band = min(uint(theta / MERIAN_M_PI * PATH_DEBUG_THETA_BANDS),
                                  PATH_DEBUG_THETA_BANDS - 1);
            const uint base = PATH_DEBUG_STATS_THETA + 3 * band;
            InterlockedAdd(stats[base + 0], maps[path_debug_map_count_at(bin)]);
            path_debug_atomic_add_float(stats, stats_f, base + 1, pdf * omega);
            path_debug_atomic_add_float(stats, stats_f, base + 2, omega);
        }
    }

    const float wave_check = WaveActiveMax(check);
    const float wave_eval = WaveActiveMax(eval_density);
    const float wave_eval_part = WaveActiveSum(eval_part);
    const float wave_tv_eval = WaveActiveSum(tv_eval);
    const float wave_tv_check = WaveActiveSum(tv_check);
    const float wave_tv_records = WaveActiveSum(tv_records);
    const float wave_density = WaveActiveMax(density);
    const float wave_contrib = WaveActiveMax(contrib);
    const float wave_mean = WaveActiveMax(mean);
    const float wave_pdf = WaveActiveMax(pdf);
    const float wave_integral = WaveActiveSum(pdf_part);
    const float wave_chi2 = WaveActiveSum(chi2);
    const uint wave_dof = WaveActiveSum(dof);
    const float wave_lag_product = WaveActiveSum(lag_product);
    const float wave_lag_square = WaveActiveSum(lag_square);
    const float wave_omega = WaveActiveSum(omega);
    if (WaveIsFirstLane()) {
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_DENSITY, wave_density);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_CHECK, wave_check);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_EVAL, wave_eval);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_EVAL_INTEGRAL, wave_eval_part);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_TV_EVAL, wave_tv_eval);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_TV_CHECK, wave_tv_check);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_TV_RECORDS, wave_tv_records);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_CONTRIB, wave_contrib);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_MEAN, wave_mean);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_PDF, wave_pdf);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_PDF_INTEGRAL, wave_integral);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_CHI2, wave_chi2);
        InterlockedAdd(stats[PATH_DEBUG_STATS_CHI2_DOF], wave_dof);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_LAG_PRODUCT, wave_lag_product);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_LAG_SQUARE, wave_lag_square);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_OMEGA, wave_omega);
    }
}
