import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.colors.colorspaces;
import merian_shaders.gbuffer;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

ParameterBlock<GBuffer> gbuffer;

uint technique_slot(const uint method) {
    return PATH_DEBUG_STATS_TECHNIQUE_CONTRIB +
           (method < PATH_RECORD_METHOD_COUNT ? method : PATH_RECORD_METHOD_UNKNOWN);
}

uint test_set(const PathRecordVertex v) {
    if (!v.nee) {
        return PATH_DEBUG_SET_SCATTER;
    }
    return v.surface.x == PATH_RECORD_GEOMETRY_ENVIRONMENT ? PATH_DEBUG_SET_NEE_ENV
                                                            : PATH_DEBUG_SET_NEE;
}

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint count = path_record_path_count(in_records);
    const uint phase =
        params.stats_stride > 1 ? pcg(in_records[PATH_RECORD_HDR_FRAME]) % params.stats_stride : 0;
    const uint i = gid.x * params.stats_stride + phase;
    if (i >= count || !path_debug_holds(i, false)) {
        return;
    }
    const PathRecordPath p = path_debug_load_selected_path(i, false);
    const uint2 selected = path_debug_path_events(i, false);
    const uint vertex_region = path_record_vertex_region_offset(in_records);

    InterlockedAdd(stats[PATH_DEBUG_STATS_COUNT], 1);
    InterlockedAdd(stats[PATH_DEBUG_STATS_LUM +
                         path_debug_log_bin(p.luminance, PATH_DEBUG_HIST_MIN_LOG,
                                            PATH_DEBUG_HIST_MAX_LOG, PATH_DEBUG_HIST_SMALL_BINS)],
                   1);

    // the directions of the selected point's draws
    const bool binning = path_debug_selects_point();
    const bool local_frame =
        merian_path_debug_gbuffer && params.map_frame == PATH_DEBUG_FRAME_LOCAL;
    const float3x3 to_map = dir_map_frame(
        local_frame
            ? path_debug_map_normal(gbuffer.get_first_hit_normal(path_debug_selected_pixel()))
            : float3(0, 0, 1),
        local_frame);

    uint chain = path_debug_chain_length(in_records, p, vertex_region);

    uint samples_acc[PATH_DEBUG_TESTED_SETS] = { 0, 0, 0 };
    uint draws_acc[PATH_DEBUG_TESTED_SETS] = { 0, 0, 0 };
    uint seen[PATH_DEBUG_TESTED_SETS] = { 0, 0, 0 };
    uint guided_acc = 0;
    float integrand_acc = 0.f;
    float3 downstream = float3(0.f);
    float3 nee_at_parent = float3(0.f);
    // a hit's light, waiting for the chain vertex whose draw reached it
    float pending_hit = 0.f;
    for (int k = int(p.vertex_count) - 1; k >= 0; k--) {
        const PathRecordVertex v =
            path_record_load_vertex(in_records, vertex_region, p.vertex_offset + k);
        const bool event = path_debug_is_event_vertex(selected, uint(k));
        const float3 contrib = event ? path_debug_finite_or_zero(v.contrib) : float3(0.f);
        const float3 transported = v.nee ? contrib : downstream;
        if (v.nee) {
            nee_at_parent += contrib;
        } else {
            downstream += contrib + nee_at_parent;
            nee_at_parent = float3(0.f);
            chain--;
            if (pending_hit > 0.f) {
                path_debug_atomic_add_float(stats, stats_f, technique_slot(v.method),
                                            pending_hit);
            }
            pending_hit = 0.f;
        }
        // the light path the vertex ends has as many scatter events as chain vertices before it
        if (event) {
            const float lum = yuv_luminance(contrib);
            const uint scatter_bin = min(chain, PATH_DEBUG_SCATTER_BINS - 1);
            InterlockedAdd(stats[PATH_DEBUG_STATS_SCATTER + scatter_bin], 1);
            path_debug_atomic_add_float(stats, stats_f,
                                        PATH_DEBUG_STATS_SCATTER_CONTRIB + scatter_bin, lum);
            if (v.nee && lum > 0.f) {
                path_debug_atomic_add_float(stats, stats_f, technique_slot(v.method), lum);
            } else if (!v.nee) {
                pending_hit = lum;
            }
        }
        const uint index = v.nee ? chain - 1 : chain;
        if (index != 0 || v.terminal) {
            continue;
        }
        const float3 uvw = dir_map_uv(params.map_transform, mul(to_map, v.wo));
        const bool in_map = binning && uvw.z != 0.f;
        const uint bin = path_debug_map_bin(path_debug_map_cell(uvw.xy));
        const float integrand = yuv_luminance(transported);
        if (in_map && integrand > 0.f) {
            path_debug_atomic_add_float3(maps, maps_f, path_debug_map_integrand_at(bin),
                                         transported);
            path_debug_atomic_add_float(maps, maps_f, path_debug_map_moment_at(bin),
                                        min(integrand * integrand, 1e30f));
            integrand_acc += integrand;
        }

        const uint set = test_set(v);
        const uint first = v.nee ? PATH_DEBUG_SET_NEE : set;
        const uint last = v.nee ? PATH_DEBUG_SET_NEE_ENV : set;
        const uint flags = (v.reused ? PATH_DEBUG_SEEN_REUSED : 0) |
                           (v.resampled ? PATH_DEBUG_SEEN_RESAMPLED : 0);
        const bool continuous =
            (v.lobe & PATH_RECORD_LOBE_DELTA) == 0 && v.pdf > 0.f && !v.resampled;
        for (uint s = first; s <= last; s++) {
            seen[s] |= flags;
            draws_acc[s] += v.resampled ? 0 : 1;
        }
        if (path_debug_is_empty_draw(v) || !in_map) {
            continue;
        }
        InterlockedAdd(maps[path_debug_map_count_at(set, bin)], 1);
        samples_acc[set]++;
        if (!v.nee && v.method == PATH_RECORD_METHOD_GUIDING) {
            InterlockedAdd(maps[path_debug_map_count_at(PATH_DEBUG_SET_GUIDING, bin)], 1);
            guided_acc++;
        }
        if (continuous) {
            InterlockedAdd(maps[path_debug_map_set_at(set, PATH_DEBUG_SET_REGION_CONTINUOUS, bin)],
                           1);
            const float recip = min(1.f / v.pdf, 1e12f);
            path_debug_atomic_add_float(
                maps, maps_f, path_debug_map_set_at(set, PATH_DEBUG_SET_REGION_RECIP, bin), recip);
            path_debug_atomic_add_float(
                maps, maps_f, path_debug_map_set_at(set, PATH_DEBUG_SET_REGION_RECIP2, bin),
                recip * recip);
        }
    }
    if (pending_hit > 0.f) {
        path_debug_atomic_add_float(stats, stats_f, technique_slot(PATH_RECORD_METHOD_UNKNOWN),
                                    pending_hit);
    }
    for (uint set = 0; set < PATH_DEBUG_TESTED_SETS; set++) {
        if (samples_acc[set] != 0) {
            InterlockedAdd(stats[path_debug_set_stat(set, PATH_DEBUG_SET_SAMPLES)],
                           samples_acc[set]);
        }
        if (draws_acc[set] != 0) {
            InterlockedAdd(stats[path_debug_set_stat(set, PATH_DEBUG_SET_DRAWS)], draws_acc[set]);
        }
        if (seen[set] != 0) {
            InterlockedOr(stats[path_debug_set_stat(set, PATH_DEBUG_SET_SEEN)], seen[set]);
        }
    }
    if (guided_acc != 0) {
        InterlockedAdd(stats[PATH_DEBUG_STATS_GUIDING_SAMPLES], guided_acc);
    }
    if (integrand_acc > 0.f) {
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_INTEGRAND, integrand_acc);
    }
}
