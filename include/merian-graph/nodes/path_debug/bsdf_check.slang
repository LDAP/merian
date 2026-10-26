import merian_shaders.scene.camera;
import merian_shaders.scene.scene;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;
import merian_shaders.utils.dir_map;
import merian_shaders.utils.frame;
import merian_shaders.shading.bsdfs.bsdf_validate;
import merian_shaders.colors.colorspaces;
import merian_shaders.gbuffer;
import merian_shaders.utils.math;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"
#include "merian-graph/nodes/path_debug/surface.slangh"

ParameterBlock<GBuffer> gbuffer;
[vk::push_constant] ConstantBuffer<PathDebugCheckPass> pass;

static const uint64_t RANDOMS_PER_THREAD = 1024;

// exact as long as the atomic rounds to nearest like the ALU
void atomic_add_compensated(const uint index, const uint error_index, const float value) {
    const float before = path_debug_atomic_add_float(maps, maps_f, index, value);
    float error;
    two_sum(before, value, error);
    if (error != 0.f) {
        path_debug_atomic_add_float(maps, maps_f, error_index, error);
    }
}

[numthreads(PATH_DEBUG_WG_2D, PATH_DEBUG_WG_2D, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID, ParameterBlock<Scene> scene) {
    const bool sampling = gid.y * PATH_DEBUG_CHECK_DIM + gid.x < pass.sample_threads;
    const bool quadrature = pass.quadrature != 0 && all(gid.xy < PATH_DEBUG_CHECK_DIM);
    if (gid.x >= PATH_DEBUG_CHECK_DIM || (!sampling && !quadrature)) {
        return;
    }
    const float3 selected_normal = gbuffer.get_first_hit_normal(path_debug_selected_pixel());
    const PathDebugSurface surface =
        path_debug_surface(scene.generate_camera_ray(path_debug_selected_pixel(), path_debug_dim()),
                           gbuffer.get_hit(path_debug_selected_pixel()), selected_normal,
                           all(gid.xy == 0));
    const RayDesc ray = surface.ray;
    const ShadingPoint sp = scene.get_shading_point(ray, surface.hit);
    if (!sp.has_surface_material) {
        return;
    }

    const float3x3 to_map = dir_map_frame(path_debug_map_normal(selected_normal),
                                          params.map_frame == PATH_DEBUG_FRAME_LOCAL);
    const OrthonormalFrame frame = sp.shading_frame;
    let bsdf = sp.surface_material.get_bsdf();
    const float3 wi_local = path_debug_wi_local(frame.to_local(-ray.Direction));

    // 1. sampled half
    const uint draws = sampling ? PATH_DEBUG_CHECK_SAMPLES_PER_CELL : 0;
    const uint64_t stream =
        (uint64_t(pass.seq) * PATH_DEBUG_CHECK_DIM + gid.y) * PATH_DEBUG_CHECK_DIM + gid.x;
    RandomGenerator rng = SplitMix64(stream * RANDOMS_PER_THREAD * 0x9E3779B97F4A7C15ull);
    uint landed = 0;
    uint valid = 0;
    uint bad = 0;
    uint transmitted = 0;
    uint singular = 0;
    uint off_weight = 0;
    uint off_pdf = 0;
    float3 weight_sum = float3(0.f);
    float transmitted_weight = 0.f;
    float continuous_weight = 0.f;
    float weight_lum_sq = 0.f;
    float err_weight = 0.f;
    float err_pdf = 0.f;
    for (uint s = 0; s < draws; s++) {
        const BSDFSampleCheck check = bsdf_validate_sample(bsdf, wi_local, rng);
        if (!check.valid) {
            continue;
        }
        valid++;
        transmitted += check.transmitted ? 1 : 0;
        singular += check.delta ? 1 : 0;
        // reported on its own, it would poison every sum below
        if (!check.sane) {
            bad++;
            continue;
        }
        err_weight = bsdf_validate_err(err_weight, check.err_weight);
        err_pdf = bsdf_validate_err(err_pdf, check.err_pdf);
        off_weight += check.err_weight > BSDF_VALIDATE_IDENTITY_TOL ? 1 : 0;
        off_pdf += check.err_pdf > BSDF_VALIDATE_IDENTITY_TOL ? 1 : 0;

        const float3 uvw = dir_map_uv(params.map_transform, mul(to_map, frame.to_world(check.wo)));
        if (uvw.z != 0.f) {
            const uint bin = path_debug_map_bin(path_debug_map_cell(uvw.xy));
            InterlockedAdd(maps[path_debug_map_check_at(bin)], 1);
            if (!check.delta) {
                const float recip = min(1.f / check.pdf, 1e12f);
                atomic_add_compensated(path_debug_map_check_recip_at(bin),
                                       path_debug_map_check_recip_error_at(bin), recip);
                atomic_add_compensated(path_debug_map_check_recip2_at(bin),
                                       path_debug_map_check_recip2_error_at(bin), recip * recip);
            }
            landed++;
        }
        const float weight_lum = yuv_luminance(check.weight);
        transmitted_weight += check.transmitted ? weight_lum : 0.f;
        continuous_weight += check.delta ? 0.f : weight_lum;
        weight_sum += check.weight;
        weight_lum_sq += weight_lum * weight_lum;
    }

    // 2. quadrature half, one full-sphere cell per thread
    const float2 uv = (float2(gid.xy) + 0.5f) / float(PATH_DEBUG_CHECK_DIM);
    const float4 cell_dir = dir_map_dir(DIR_MAP_OCT, uv);
    const float solid_angle = dir_map_bin_solid_angle(DIR_MAP_OCT, gid.xy, PATH_DEBUG_CHECK_DIM);
    float pdf_part = 0.f;
    float3 eval_part = float3(0.f);
    float err_reciprocity = 0.f;
    uint off_reciprocity = 0;
    uint reciprocity_cells = 0;
    uint bad_cells = 0;
    if (quadrature && cell_dir.w != 0.f && solid_angle > 0.f) {
        const BSDFDirectionCheck check = bsdf_validate_direction(bsdf, wi_local, cell_dir.xyz, rng);
        pdf_part = check.pdf * solid_angle;
        eval_part = check.eval * solid_angle;
        err_reciprocity = check.err_reciprocity;
        bad_cells = check.sane ? 0 : 1;
        if (cell_dir.z * wi_local.z > 0.f) {
            reciprocity_cells = 1;
            off_reciprocity = check.err_reciprocity > BSDF_VALIDATE_IDENTITY_TOL ? 1 : 0;
        }
    }

    // 3. one atomic per wave and quantity
    const uint wave_draws = WaveActiveSum(draws);
    const uint wave_cells = WaveActiveSum(quadrature ? 1 : 0);
    const uint wave_landed = WaveActiveSum(landed);
    const uint wave_valid = WaveActiveSum(valid);
    const uint wave_bad = WaveActiveSum(bad);
    const uint wave_transmitted = WaveActiveSum(transmitted);
    const uint wave_singular = WaveActiveSum(singular);
    const uint wave_off_weight = WaveActiveSum(off_weight);
    const uint wave_off_pdf = WaveActiveSum(off_pdf);
    const uint wave_off_reciprocity = WaveActiveSum(off_reciprocity);
    const uint wave_reciprocity_cells = WaveActiveSum(reciprocity_cells);
    const uint wave_bad_cells = WaveActiveSum(bad_cells);
    const float3 wave_weight = WaveActiveSum(weight_sum);
    const float wave_weight_lum = yuv_luminance(wave_weight);
    const float wave_mean_lum = wave_weight_lum / float(max(wave_draws, 1u));
    // within-wave squared deviations bound the cancellation of sum(x^2) - n mean^2
    const float wave_weight_m2 =
        max(WaveActiveSum(weight_lum_sq) - wave_weight_lum * wave_mean_lum, 0.f);
    const float wave_transmitted_weight = WaveActiveSum(transmitted_weight);
    const float wave_continuous_weight = WaveActiveSum(continuous_weight);
    const float wave_pdf = WaveActiveSum(pdf_part);
    const float3 wave_eval = WaveActiveSum(eval_part);
    const float wave_err_weight = WaveActiveMax(err_weight);
    const float wave_err_pdf = WaveActiveMax(err_pdf);
    const float wave_err_reciprocity = WaveActiveMax(err_reciprocity);
    if (WaveIsFirstLane() && wave_draws != 0) {
        if (wave_landed != 0) {
            InterlockedAdd(stats[PATH_DEBUG_STATS_CHECK_SAMPLES], wave_landed);
        }
        InterlockedAdd(stats[PATH_DEBUG_CHECK_DRAWS], wave_draws);
        InterlockedAdd(stats[PATH_DEBUG_CHECK_VALID], wave_valid);
        InterlockedAdd(stats[PATH_DEBUG_CHECK_BAD], wave_bad);
        InterlockedAdd(stats[PATH_DEBUG_CHECK_TRANSMITTED], wave_transmitted);
        if (wave_singular != 0) {
            InterlockedAdd(stats[PATH_DEBUG_CHECK_SINGULAR], wave_singular);
        }
        InterlockedAdd(stats[PATH_DEBUG_CHECK_OFF_WEIGHT], wave_off_weight);
        InterlockedAdd(stats[PATH_DEBUG_CHECK_OFF_PDF], wave_off_pdf);
        path_debug_atomic_add_float3(stats, stats_f, PATH_DEBUG_CHECK_WEIGHT, wave_weight);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_CHECK_WEIGHT_M2, wave_weight_m2);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_CHECK_TRANSMITTED_WEIGHT,
                                    wave_transmitted_weight);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_CHECK_CONTINUOUS_WEIGHT,
                                    wave_continuous_weight);
        path_debug_atomic_max_float(stats, PATH_DEBUG_CHECK_ERR_WEIGHT, wave_err_weight);
        path_debug_atomic_max_float(stats, PATH_DEBUG_CHECK_ERR_PDF, wave_err_pdf);
    }
    if (WaveIsFirstLane() && wave_cells != 0) {
        InterlockedAdd(stats[PATH_DEBUG_CHECK_CELLS], wave_cells);
        InterlockedAdd(stats[PATH_DEBUG_CHECK_OFF_RECIPROCITY], wave_off_reciprocity);
        InterlockedAdd(stats[PATH_DEBUG_CHECK_RECIPROCITY_CELLS], wave_reciprocity_cells);
        InterlockedAdd(stats[PATH_DEBUG_CHECK_BAD_CELLS], wave_bad_cells);
        path_debug_atomic_add_float3(stats, stats_f, PATH_DEBUG_CHECK_EVAL_INTEGRAL, wave_eval);
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_CHECK_PDF_INTEGRAL, wave_pdf);
        path_debug_atomic_max_float(stats, PATH_DEBUG_CHECK_ERR_RECIPROCITY, wave_err_reciprocity);
    }
    if (quadrature && all(gid.xy == 0)) {
        path_debug_store_float3(stats, PATH_DEBUG_CHECK_ALBEDO,
                                bsdf.get_albedo(wi_local).reflection);
        stats[PATH_DEBUG_CHECK_ALPHA] = asuint(bsdf.effective_alpha);
    }
}
