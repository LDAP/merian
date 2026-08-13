import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.utils.encoding;
import merian_shaders.utils.frame;
import merian_shaders.colors.colorspaces;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

// Display maxima and the pdf sanity integral over the directional maps; the slots are zeroed by
// finalize at the end of the previous frame.
[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint bin = gid.x;
    float density = 0.f;
    float contrib = 0.f;
    float mean = 0.f;
    float pdf = 0.f;
    float pdf_part = 0.f;
    if (bin < params.map_res * params.map_res) {
        const uint2 cell = uint2(bin % params.map_res, bin / params.map_res);
        const float solid_angle = dir_map_bin_solid_angle(params.map_transform, cell, params.map_res);
        const uint total = max(stats[PATH_DEBUG_STATS_SAMPLES], 1u);
        const uint count = maps[path_debug_map_count_at(bin)];
        const uint c = path_debug_map_contrib_at(bin);
        const float lum = yuv_luminance(
            float3(asfloat(maps[c + 0]), asfloat(maps[c + 1]), asfloat(maps[c + 2])));
        pdf = abs(asfloat(maps[path_debug_map_pdf_at(bin)]));
        if (solid_angle > 0.f) {
            density = float(count) / (float(total) * solid_angle);
            contrib = lum / (float(total) * solid_angle);
            pdf_part = pdf * solid_angle;
        }
        mean = count > 0 ? lum / float(count) : 0.f;
    }

    // per-bin consistency statistic against the selected reference
    float chi2 = 0.f;
    uint dof = 0;
    float omega = 0.f;
    if (bin < params.map_res * params.map_res) {
        const uint2 cell = uint2(bin % params.map_res, bin / params.map_res);
        const float solid_angle = dir_map_bin_solid_angle(params.map_transform, cell, params.map_res);
        omega = solid_angle;
        const float count = float(maps[path_debug_map_count_at(bin)]);
        if (params.map_reference == PATH_DEBUG_REFERENCE_RECORDED) {
            // reciprocal-pdf identity: E[sum 1/pdf] = draws * omega per bin, valid where every
            // draw shares one support (one surface: selected pixel, bounce 0); the second
            // reciprocal moment estimates the variance
            const float sum_recip = asfloat(maps[path_debug_map_recip_at(bin)]);
            const float sum_recip2 = asfloat(maps[path_debug_map_recip2_at(bin)]);
            const float expected = float(stats[PATH_DEBUG_STATS_DRAWS]) * solid_angle;
            const bool shared_support = params.map_scope == 0 && params.map_bounce == 0;
            if (shared_support && sum_recip2 > 0.f && count >= 5.f && expected >= 1.f) {
                const float z = (sum_recip - expected) / sqrt(sum_recip2);
                chi2 = z * z;
                dof = 1;
                InterlockedAdd(stats[PATH_DEBUG_STATS_TESTED], uint(count));
            }
        } else {
            // Poisson counts against the BSDF pdf, conditioned on the pdf mass the map covers
            // (previous frame's integral): the samples are all inside the mapped domain, the
            // raw pdf may not be
            const bool resolved = asfloat(maps[path_debug_map_pdf_at(bin)]) >= 0.f;
            const float norm = max(asfloat(stats[PATH_DEBUG_STATS_PDF_NORM]), 1e-6f);
            const float expected =
                float(stats[PATH_DEBUG_STATS_SAMPLES]) * pdf * solid_angle / norm;
            // a needle-sharp lobe (quasi-delta) is invisible to the sub-sampled pdf: counts
            // vastly above a smooth (possibly near-zero) expectation are separated out instead
            // of read as bias
            const bool spike = count >= 16.f && count > 10.f * expected;
            if (spike) {
                path_debug_atomic_add_float(stats, PATH_DEBUG_STATS_SPIKE_MASS,
                                            (count - expected) /
                                                max(float(stats[PATH_DEBUG_STATS_SAMPLES]), 1.f));
                InterlockedAdd(stats[PATH_DEBUG_STATS_SPIKE_BINS], 1);
            }
            if (resolved && !spike && expected >= 5.f) {
                const float diff = count - expected;
                chi2 = diff * diff / expected;
                dof = 1;
                InterlockedAdd(stats[PATH_DEBUG_STATS_TESTED], uint(count));
            }
        }
    }

    // polar profile bands for the overlaid density-vs-pdf curves
    if (bin < params.map_res * params.map_res && omega > 0.f) {
        const uint2 cell = uint2(bin % params.map_res, bin / params.map_res);
        const float2 uv = (float2(cell) + 0.5f) / float(params.map_res);
        const float4 center = dir_map_dir(params.map_transform, uv);
        if (center.w != 0.f) {
            const float theta = acos(clamp(center.z, -1.f, 1.f));
            const uint band = min(uint(theta / 3.14159265f * PATH_DEBUG_THETA_BANDS),
                                  PATH_DEBUG_THETA_BANDS - 1);
            const uint base = PATH_DEBUG_STATS_THETA + 3 * band;
            InterlockedAdd(stats[base + 0], maps[path_debug_map_count_at(bin)]);
            path_debug_atomic_add_float(stats, base + 1, pdf * omega);
            path_debug_atomic_add_float(stats, base + 2, omega);
        }
    }

    const float wave_density = WaveActiveMax(density);
    const float wave_contrib = WaveActiveMax(contrib);
    const float wave_mean = WaveActiveMax(mean);
    const float wave_pdf = WaveActiveMax(pdf);
    const float wave_integral = WaveActiveSum(pdf_part);
    const float wave_chi2 = WaveActiveSum(chi2);
    const uint wave_dof = WaveActiveSum(dof);
    const float wave_omega = WaveActiveSum(omega);
    if (WaveIsFirstLane()) {
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_DENSITY, wave_density);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_CONTRIB, wave_contrib);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_MEAN, wave_mean);
        path_debug_atomic_max_float(stats, PATH_DEBUG_STATS_MAX_PDF, wave_pdf);
        path_debug_atomic_add_float(stats, PATH_DEBUG_STATS_PDF_INTEGRAL, wave_integral);
        path_debug_atomic_add_float(stats, PATH_DEBUG_STATS_CHI2, wave_chi2);
        InterlockedAdd(stats[PATH_DEBUG_STATS_CHI2_DOF], wave_dof);
        path_debug_atomic_add_float(stats, PATH_DEBUG_STATS_OMEGA, wave_omega);
    }
}
