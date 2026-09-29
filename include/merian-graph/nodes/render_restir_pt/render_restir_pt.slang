#include "merian-graph/nodes/render_restir_pt/restir_pt_neighbors.slangh"
#include "merian-graph/nodes/render_restir_pt/restir_pt_record.slangh"
#include "merian-graph/nodes/render_restir_pt/restir_pt_shift.slangh"

using merian;

namespace merian {
extern static const int merian_restir_pt_spp = 1;
extern static const bool merian_restir_pt_demodulate_albedo = false;
// backproject to one of the four pixels around the motion vector's end, in proportion to overlap
extern static const bool merian_restir_pt_stochastic_backprojection = false;
// follow the occluder out of a disocclusion [Zeng et al. 2021]
extern static const bool merian_restir_pt_disocclusion_motion = false;
// lower the history cap where the previous frame's path repeats around the pixel
// [Lin et al. 2026, 6]
extern static const bool merian_restir_pt_duplication_cap = false;
// shade with every resampled path by its resampling weight rather than with the one selected
extern static const bool merian_restir_pt_decoupled_shading = false;
extern static const uint merian_restir_pt_temporal_mode = RestirPTTemporalBackprojection;
// backproject what a smooth primary surface reflects by the motion of its virtual image
extern static const bool merian_restir_pt_specular_motion = false;
extern static const float merian_restir_pt_specular_motion_roughness = 0.1f;
extern static const uint merian_restir_pt_debug_view = RestirPTDebugNone;
extern static const uint merian_restir_pt_record = RestirPTRecordCandidates;
} // namespace merian

struct RestirPTBinding {
    ParameterBlock<GBuffer> gbuffer;
    ParameterBlock<GBuffer> prev_gbuffer;
    RWTexture2D<float4> irradiance;
    // RestirPTQueueHeader followed by the queued shifts
    RWStructuredBuffer<uint> queue;
    RWStructuredBuffer<uint, ScalarDataLayout> path_records;
};

[vk::push_constant] ConstantBuffer<RestirPTPushConstant> pc;

static const float RESTIR_PT_DUPLICATION_WINDOW =
    float((2 * RESTIR_PT_DUPLICATION_RADIUS + 1) * (2 * RESTIR_PT_DUPLICATION_RADIUS + 1) - 1);

uint restir_pt_index(const uint2 pixel, const uint2 dim) {
    return pixel.y * dim.x + pixel.x;
}

uint restir_pt_slot(const uint index, const uint slot) {
    return index * RESTIR_PT_MAX_NEIGHBORS + slot;
}

uint restir_pt_shift_index(const uint index, const uint slot, const uint direction) {
    return restir_pt_slot(index, slot) * 2 + direction;
}

RandomGenerator restir_pt_rng(const uint2 pixel, const RestirPTPass pass) {
    return PCG(xxhash32(uint4(pixel, pc.frame * RestirPTPassCount + pass, pc.seed + pc.round)));
}

bool restir_pt_in_screen(const int2 pixel, const uint2 dim) {
    return all(pixel >= 0) && all(pixel < int2(dim));
}

// Appends this lane's shifts to the queue, one atomic per wave, and grows the indirect dispatch
// to cover them. Every lane of the dispatch has to arrive here.
void restir_pt_enqueue(const RestirPTBinding params,
                       const uint items[2 * RESTIR_PT_MAX_NEIGHBORS],
                       const uint count) {
    const uint total = WaveActiveSum(count);
    const uint prefix = WavePrefixSum(count);
    uint base = 0;
    if (WaveIsFirstLane() && total > 0) {
        InterlockedAdd(params.queue[3], total, base);
        InterlockedMax(params.queue[0], (base + total + 63) / 64);
    }
    base = WaveReadLaneFirst(base);
    for (uint i = 0; i < count; i++)
        params.queue[4 + base + prefix + i] = items[i];
}

uint restir_pt_item(const uint index, const uint slot, const uint direction) {
    return (index << 4) | (slot << 1) | direction;
}

static const int2 RESTIR_PT_COVERAGE_OFFSETS[8] = {int2(-1, -1), int2(0, -1), int2(1, -1),
                                                   int2(-1, 0),  int2(1, 0),  int2(-1, 1),
                                                   int2(0, 1),   int2(1, 1)};

// Where the shift of a previous reservoir into its neighbor at `offset` is kept.
uint restir_pt_coverage_shift_index(const uint index, const int2 offset) {
    const int k = 3 * (offset.y + 1) + offset.x + 1;
    return index * 2 * RESTIR_PT_MAX_NEIGHBORS + uint(k < 4 ? k : k - 1);
}

bool restir_pt_subpixel_backprojection() {
    return merian_restir_pt_area &&
           (merian_restir_pt_temporal_mode == RestirPTTemporalSubpixelAsIs ||
            merian_restir_pt_temporal_mode == RestirPTTemporalSubpixelCoverage);
}

bool restir_pt_splatting() {
    return merian_restir_pt_temporal_mode == RestirPTTemporalSplatting ||
           merian_restir_pt_temporal_mode == RestirPTTemporalSplattingFallback;
}

// Top-left corner of the pixel's footprint backprojected into the previous image.
float2 restir_pt_backprojected_origin(const uint index) {
    return asfloat(uint2(pc.temporal_domains[4 * index], pc.temporal_domains[4 * index + 1]));
}

// Writes the image from the reservoir that is final this frame, on top of the primary emission the
// initial pass left there.
void restir_pt_resolve(const RestirPTBinding params,
                       const uint2 pixel,
                       const RestirPTReservoir r,
                       const float3 decoupled) {
    float3 contribution = merian_restir_pt_decoupled_shading ? decoupled : r.integrand * r.W;
    if (!all(isfinite(contribution)))
        contribution = float3(0);
    float3 color = params.irradiance[pixel].rgb + contribution;
    if (merian_restir_pt_demodulate_albedo)
        color /= params.gbuffer.get_demodulation_albedo(pixel);

    if (merian_restir_pt_debug_view == RestirPTDebugReconnectionVertex) {
        const float n = float(restir_pt_light_vertex(r.flags));
        const float k = float(restir_pt_rc_vertex(r.flags));
        color = r.W > 0.f ? float3(k / max(n, 1.f), k == 0.f ? 1.f : 0.f, n / 16.f) : float3(0);
    } else if (merian_restir_pt_debug_view == RestirPTDebugConfidence) {
        color = float3(r.M / pc.history_cap);
    }
    const float l = yuv_luminance(color);
    params.irradiance[pixel] = float4(color, l * l);
}

void restir_pt_debug_reuse(const RestirPTBinding params,
                           const uint2 pixel,
                           const uint candidates,
                           const uint accepted,
                           const uint shifted,
                           const bool neighbor_won) {
    if (merian_restir_pt_debug_view != RestirPTDebugReuseSuccess ||
        (pc.flags & RestirPTPassResolves) == 0)
        return;
    params.irradiance[pixel] =
        float4(float(accepted) / float(max(candidates, 1u)),
               float(shifted) / float(max(accepted, 1u)), neighbor_won ? 1.f : 0.f, 0.f);
}

// One path per sample, resampled into a single reservoir. The samples are independent, so each
// enters with m_i = 1 / spp. The reservoir set no pass holds yet keeps the running reconnection
// vertex of the path being traced.
void restir_pt_pass_initial(const uint2 pixel,
                            const ParameterBlock<Scene> scene,
                            const ParameterBlock<RestirPTBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    RandomGenerator rng = restir_pt_rng(pixel, RestirPTPassInitial);
    const uint index = restir_pt_index(pixel, dim);
    RestirPTCandidates candidates;
    candidates.selected = pc.reservoirs_out + index;
    candidates.running = pc.reservoirs_in + index;
    candidates.weight_sum = 0.f;
    float3 primary_emission = float3(0);
    for (int s = 0; s < merian_restir_pt_spp; s++) {
        // zero marks "no path" in the duplication map
        candidates.seed = max(xxhash32(uint4(pixel, pc.frame, uint(s) ^ (pc.seed << 8))), 1u);
        const RestirPTPrimary primary = restir_pt_primary(
            scene, params.gbuffer, false,
            float2(pixel) + restir_pt_unpack_unorm(RestirPTRandom(candidates.seed).film()), dim,
            candidates.seed);
        var rec = PathRecorder::disabled();
        if (merian_restir_pt_record == RestirPTRecordCandidates)
            rec = PathRecorder::begin(params.path_records, pixel, uint(s), pc.frame,
                                      2 * uint(merian_restir_pt_max_path_length) + 1);
        if (primary.valid)
            primary_emission += restir_pt_trace_candidates(scene, primary, candidates, rng,
                                                           params.path_records, rec);
        else
            rec.finish(params.path_records);
    }

    RestirPTReservoir r = restir_pt_empty_reservoir();
    if (candidates.weight_sum > 0.f) {
        r = pc.reservoirs_out[index];
        const float target = yuv_luminance(r.integrand);
        r.W = target > 0.f ? candidates.weight_sum / (float(merian_restir_pt_spp) * target) : 0.f;
    }
    r.M = 1.f;
    pc.reservoirs_out[index] = r;
    const float3 emission = primary_emission / float(merian_restir_pt_spp);
    params.irradiance[pixel] = float4(all(isfinite(emission)) ? emission : float3(0), 0);
    if ((pc.flags & RestirPTPassResolves) != 0)
        restir_pt_resolve(params, pixel, r, r.integrand * r.W);
}

// The path each pixel shades this frame, replayed in its own domain.
void restir_pt_pass_record(const uint2 pixel,
                           const ParameterBlock<Scene> scene,
                           const ParameterBlock<RestirPTBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    if (!merian_path_record || any(pixel >= dim))
        return;
    const RestirPTReservoir r = pc.reservoirs_in[restir_pt_index(pixel, dim)];
    var rec = PathRecorder::begin(params.path_records, pixel, 0, pc.frame,
                                  uint(merian_restir_pt_max_path_length) + 2);
    const RestirPTPrimary primary = restir_pt_primary(
        scene, params.gbuffer, false, restir_pt_image_pos(float2(pixel), r), dim, r.seed);
    RandomGenerator rng = restir_pt_rng(pixel, RestirPTPassRecord);
    if (primary.valid)
        restir_pt_record_shaded(scene, primary, r, params.path_records, rec, rng);
    else
        rec.finish(params.path_records);
}

// The history cap of the reservoir at a previous pixel, lowered where the previous frame's path
// repeats around it [Lin et al. 2026, 6].
float restir_pt_history_cap(const uint prev_index) {
    if (!merian_restir_pt_duplication_cap)
        return pc.history_cap;
    const float duplication =
        float(pc.prev_duplication[prev_index] & 0xFFFFu) / RESTIR_PT_DUPLICATION_WINDOW;
    return lerp(pc.history_cap, 1.f, pow(duplication, pc.duplication_exponent));
}

// The history reservoir of a slot, capped, with its suffix as it arrives now.
RestirPTReservoir restir_pt_history(const ParameterBlock<Scene> scene,
                                    const uint prev_index,
                                    const uint index,
                                    const uint slot) {
    RestirPTReservoir history;
    if (restir_pt_subpixel_backprojection()) {
        history = pc.backprojected[index];
    } else {
        history = pc.reservoirs_prev[prev_index];
        restir_pt_from_prev_frame(scene, history);
    }
    history.M = min(history.M, restir_pt_history_cap(prev_index));
    if (merian_restir_pt_dynamic_scene && history.W > 0.f) {
        const RestirPTSuffix suffix = pc.history[restir_pt_slot(index, slot)];
        history.rc_wo = suffix.rc_wo;
        history.light_pdf = suffix.light_pdf;
        history.rc_radiance = suffix.rc_radiance;
    }
    return history;
}

// The previous pixel this pixel's motion vector points at.
int2 restir_pt_backprojected_pixel(const uint2 pixel,
                                   const uint2 dim,
                                   const ParameterBlock<Scene> scene,
                                   const ParameterBlock<RestirPTBinding> params,
                                   inout RandomGenerator rng) {
    float2 motion = params.gbuffer.get_motion_vectors(pixel);
    if (merian_restir_pt_specular_motion) {
        // a smooth surface shows its specular chain's end as a virtual image behind it, and the
        // specular share of the paths leave with it
        const float chain = params.gbuffer.get_specular_hit_distance(pixel);
        const float specular = yuv_luminance(params.gbuffer.get_specular_albedo(pixel));
        const float diffuse = yuv_luminance(params.gbuffer.get_diffuse_albedo(pixel));
        if (chain > 0.f &&
            params.gbuffer.get_roughness(pixel) <= merian_restir_pt_specular_motion_roughness &&
            rng.next_float() * (specular + diffuse) < specular) {
            const float3 image = scene.camera.position_from_linear_z(
                pixel, dim, params.gbuffer.get_linear_z(pixel) + chain);
            motion = scene.compute_motion_vector(image, image, pixel, dim);
        }
    }
    if (merian_restir_pt_disocclusion_motion && params.gbuffer.get_hit(pixel).is_valid()) {
        const int2 prev = int2(round(float2(pixel) + motion));
        if (!restir_pt_in_screen(prev, dim)) {
            motion = -motion;
        } else {
            // the pixel was hidden behind something closer: follow that instead
            Hit occluder = params.prev_gbuffer.get_hit(uint2(prev));
            const float dist = params.gbuffer.get_linear_z(pixel);
            const float prev_dist = params.prev_gbuffer.get_linear_z(uint2(prev));
            if (occluder.is_valid() && scene.from_prev_frame(occluder) &&
                restir_pt_hit_exists(scene, occluder) && dist - prev_dist > 0.1f * dist) {
                const float3 now =
                    scene.get_surface(occluder, scene.get_triangle_hit(occluder)).pos;
                motion = float2(prev) - scene.camera.project_pinhole(now, dim);
            }
        }
    }
    const float2 offset = merian_restir_pt_stochastic_backprojection ? rng.next_float2() : 0.5f;
    return int2(floor(float2(pixel) + motion + offset));
}

// Area reservoirs splat without the backprojection fallback: which of the two a pixel took would
// depend on where other samples landed, and its MIS weights with it.
bool restir_pt_subpixel_splatting() {
    return merian_restir_pt_area && restir_pt_splatting();
}

// With area reservoirs a splat lands on the point of the image the primary vertex projects to and
// carries the Jacobian between the film areas, and every current path splats back into the
// previous frame to find the domain its MIS weight is taken in [Liu et al. 2025].
void restir_pt_pass_splat_area(const uint2 pixel,
                               const ParameterBlock<Scene> scene,
                               const ParameterBlock<RestirPTBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    const uint index = restir_pt_index(pixel, dim);

    // 1. the previous path of this pixel forward
    const RestirPTReservoir prev = pc.reservoirs_prev[index];
    if (prev.W > 0.f) {
        const RestirPTPrimary primary =
            restir_pt_primary(scene, params.prev_gbuffer, true,
                              restir_pt_image_pos(float2(pixel), prev), dim, prev.seed);
        if (primary.valid) {
            if (let splat = restir_pt_splat(scene, primary, false,
                                            params.gbuffer.get_instance_mask(), dim, prev.seed)) {
                const uint2 target = uint2(splat.image_pos);
                const uint target_index = restir_pt_index(target, dim);
                uint slot;
                InterlockedAdd(pc.splat_counts[target_index], 1u, slot);
                if (slot < RESTIR_PT_MAX_NEIGHBORS) {
                    const uint s = restir_pt_slot(target_index, slot);
                    pc.neighbors[s] = restir_pt_pack_pixel(int2(pixel));
                    pc.splat_samples[2 * s] =
                        restir_pt_pack_unorm(splat.image_pos - float2(target));
                    pc.splat_samples[2 * s + 1] = asuint(splat.jacobian);
                }
            }
        }
    }

    // 2. the current path of this pixel backward, with the confidence of where it lands
    const RestirPTReservoir current = pc.reservoirs_in[index];
    uint4 domain = uint4(RESTIR_PT_NO_NEIGHBOR, 0, 0, 0);
    if (current.W > 0.f) {
        const RestirPTPrimary primary =
            restir_pt_primary(scene, params.gbuffer, false,
                              restir_pt_image_pos(float2(pixel), current), dim, current.seed);
        if (primary.valid) {
            if (let splat =
                    restir_pt_splat(scene, primary, true, params.prev_gbuffer.get_instance_mask(),
                                    dim, current.seed)) {
                const uint2 target = uint2(splat.image_pos);
                const uint target_index = restir_pt_index(target, dim);
                const float confidence =
                    min(pc.reservoirs_prev[target_index].M, restir_pt_history_cap(target_index));
                domain = uint4(restir_pt_pack_pixel(int2(target)),
                               restir_pt_pack_unorm(splat.image_pos - float2(target)),
                               asuint(splat.jacobian), asuint(confidence));
            }
        }
    }
    for (uint i = 0; i < 4; i++)
        pc.temporal_domains[4 * index + i] = domain[i];
}

// Pushes the previous reservoir of every pixel onto the pixel its primary vertex projects to now
// [Liu et al. 2025], which follows the vertex rather than a screen-space motion vector. A pixel
// keeps the first RESTIR_PT_MAX_NEIGHBORS that land on it.
void restir_pt_pass_splat(const uint2 pixel,
                          const ParameterBlock<Scene> scene,
                          const ParameterBlock<RestirPTBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;
    if (merian_restir_pt_area) {
        restir_pt_pass_splat_area(pixel, scene, params);
        return;
    }
    Hit hit = params.prev_gbuffer.get_hit(pixel);
    if (!hit.is_valid() || !scene.from_prev_frame(hit) || !restir_pt_hit_exists(scene, hit))
        return;
    const float3 pos = scene.get_surface(hit, scene.get_triangle_hit(hit)).pos;
    const float dist = distance(pos, scene.camera.position);
    if (!(dot(pos - scene.camera.position, scene.camera.W) > 0.f))
        return;
    const int2 target = int2(round(scene.camera.project_pinhole(pos, dim)));
    if (!restir_pt_in_screen(target, dim))
        return;
    // landing behind what the pixel sees now is landing on something else
    const float target_dist = params.gbuffer.get_linear_z(uint2(target));
    if (abs(dist - target_dist) >= merian_restir_pt_reject_depth * target_dist)
        return;
    const uint target_index = restir_pt_index(uint2(target), dim);
    uint slot;
    InterlockedAdd(pc.splat_counts[target_index], 1u, slot);
    if (slot < RESTIR_PT_MAX_NEIGHBORS)
        pc.neighbors[restir_pt_slot(target_index, slot)] = restir_pt_pack_pixel(int2(pixel));
}

// Queues the one-pixel shifts of every previous reservoir into its eight neighbors, which
// guaranteed coverage weighs its samples with [Zhang et al. 2024, 4.3.2].
void restir_pt_pass_coverage_neighbors(const uint2 thread_pixel,
                                       const ParameterBlock<RestirPTBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    const uint2 pixel = min(thread_pixel, dim - 1);
    const uint index = restir_pt_index(pixel, dim);
    uint items[2 * RESTIR_PT_MAX_NEIGHBORS];
    uint count = 0;
    if (all(thread_pixel < dim) && pc.reservoirs_prev[index].W > 0.f) {
        for (uint k = 0; k < 8; k++) {
            if (restir_pt_in_screen(int2(pixel) + RESTIR_PT_COVERAGE_OFFSETS[k], dim))
                items[count++] = restir_pt_item(index, k, 0);
        }
    }
    restir_pt_enqueue(params, items, count);
}

// The reservoir of the pixel's footprint backprojected into the previous image: the previous
// samples of the up to four pixels it overlaps that fall into it, relabeled to it, with the
// overlap-weighted confidence of those pixels [Zhang et al. 2024, 4.3.1]. With guaranteed coverage,
// the samples that fall outside move in by a one-pixel shift, weighed against the other overlapped
// pixels [Zhang et al. 2024, 4.3.2].
void restir_pt_pass_subpixel_backprojection(const uint2 pixel,
                                            const ParameterBlock<Scene> scene,
                                            const ParameterBlock<RestirPTBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;
    const uint index = restir_pt_index(pixel, dim);
    RandomGenerator rng = restir_pt_rng(pixel, RestirPTPassSubpixelBackprojection);
    const bool coverage = merian_restir_pt_temporal_mode == RestirPTTemporalSubpixelCoverage;
    const float2 origin = float2(pixel) + params.gbuffer.get_motion_vectors(pixel);
    pc.temporal_domains[4 * index] = asuint(origin.x);
    pc.temporal_domains[4 * index + 1] = asuint(origin.y);

    RestirPTReservoir selected = restir_pt_empty_reservoir();
    RestirPTShift selected_shift = {float3(0), 1.f};
    float weight_sum = 0.f;
    float confidence = 0.f;
    if (all(origin > -1.f) && all(origin < float2(dim) + 1.f)) {
        const int2 quad = int2(floor(origin));
        const float2 a = origin - float2(quad);
        float quad_confidence[4];
        for (uint c = 0; c < 4; c++) {
            const int2 q = quad + int2(c & 1, c >> 1);
            const float2 cover = lerp(1.f - a, a, float2(c & 1, c >> 1));
            quad_confidence[c] =
                restir_pt_in_screen(q, dim)
                    ? cover.x * cover.y * pc.reservoirs_prev[restir_pt_index(uint2(q), dim)].M
                    : 0.f;
            confidence += quad_confidence[c];
        }
        for (uint c = 0; c < 4; c++) {
            const int2 corner = int2(c & 1, c >> 1);
            const int2 q = quad + corner;
            if (!restir_pt_in_screen(q, dim))
                continue;
            const uint q_index = restir_pt_index(uint2(q), dim);
            RestirPTReservoir r = pc.reservoirs_prev[q_index];
            restir_pt_from_prev_frame(scene, r);
            const float target = yuv_luminance(r.integrand);
            if (!(r.W > 0.f) || !(target > 0.f))
                continue;
            const float2 uv = float2(corner) + restir_pt_unpack_unorm(r.film);
            int2 moved = int2(0);
            float weight;
            RestirPTShift shift = {r.integrand, 1.f};
            if (!coverage) {
                if (any(uv < a) || any(uv >= a + 1.f))
                    continue;
                weight = target * r.W;
            } else {
                moved = int2(uv < a) - int2(uv > a + 1.f);
                if (any(moved != 0) && !restir_pt_in_screen(q + moved, dim))
                    continue;
                if (any(moved != 0))
                    shift = pc.shifts[restir_pt_coverage_shift_index(q_index, moved)];
                // the sample's density in every overlapped pixel, by overlap and confidence
                float density_sum = quad_confidence[c] * target;
                for (uint o = 0; o < 4; o++) {
                    const int2 offset = int2(o & 1, o >> 1) - corner;
                    if (o == c || quad_confidence[o] == 0.f)
                        continue;
                    const RestirPTShift there =
                        pc.shifts[restir_pt_coverage_shift_index(q_index, offset)];
                    density_sum +=
                        quad_confidence[o] * yuv_luminance(there.integrand) * there.jacobian;
                }
                weight = yuv_luminance(shift.integrand) * shift.jacobian * r.W *
                         quad_confidence[c] * target / density_sum;
            }
            if (!(weight > 0.f) || !isfinite(weight))
                continue;
            weight_sum += weight;
            if (rng.next_float() * weight_sum < weight) {
                r.film = restir_pt_pack_unorm(saturate(uv + float2(moved) - a));
                selected = r;
                selected_shift = shift;
            }
        }
    }
    selected.integrand = selected_shift.integrand;
    selected.rc_geometry *= selected_shift.jacobian;
    const float target = yuv_luminance(selected.integrand);
    selected.W = target > 0.f ? weight_sum / target : 0.f;
    selected.M = confidence;
    pc.backprojected[index] = selected;
}

// Pairs every pixel with the previous reservoirs that splatted onto it, or with the one its motion
// vector points at, sums their confidence, and queues the shifts in both directions.
void restir_pt_pass_temporal_neighbors(const uint2 thread_pixel,
                                       const ParameterBlock<Scene> scene,
                                       const ParameterBlock<RestirPTBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    const uint2 pixel = min(thread_pixel, dim - 1);
    const bool active = all(thread_pixel < dim);
    const uint index = restir_pt_index(pixel, dim);
    RandomGenerator rng = restir_pt_rng(pixel, RestirPTPassTemporalNeighbors);
    const RestirPTReservoir canonical = pc.reservoirs_in[index];

    if (restir_pt_subpixel_backprojection()) {
        // the backprojected footprint's reservoir, with the previous pixel nearest to it for the
        // history cap
        const RestirPTReservoir backprojected = pc.backprojected[index];
        const int2 nearest = clamp(int2(floor(restir_pt_backprojected_origin(index) + 0.5f)),
                                   int2(0), int2(dim) - 1);
        const bool valid = all(thread_pixel < dim) && backprojected.M > 0.f;
        uint items[2 * RESTIR_PT_MAX_NEIGHBORS];
        uint count = 0;
        if (valid && backprojected.W > 0.f)
            items[count++] = restir_pt_item(index, 0, 0);
        if (valid && canonical.W > 0.f)
            items[count++] = restir_pt_item(index, 0, 1);
        if (active) {
            pc.neighbors[restir_pt_slot(index, 0)] =
                valid ? restir_pt_pack_pixel(nearest) : RESTIR_PT_NO_NEIGHBOR;
            for (uint s = 1; s < RESTIR_PT_MAX_NEIGHBORS; s++)
                pc.neighbors[restir_pt_slot(index, s)] = RESTIR_PT_NO_NEIGHBOR;
        }
        restir_pt_enqueue(params, items, count);
        return;
    }

    uint landed = 0;
    if (restir_pt_splatting())
        landed = min(pc.splat_counts[index], RESTIR_PT_MAX_NEIGHBORS);
    const bool backproject = !restir_pt_splatting() ||
                             (merian_restir_pt_temporal_mode == RestirPTTemporalSplattingFallback &&
                              landed == 0 && !merian_restir_pt_area);
    if (backproject && active) {
        pc.neighbors[restir_pt_slot(index, 0)] =
            restir_pt_pack_pixel(restir_pt_backprojected_pixel(pixel, dim, scene, params, rng));
        landed = 1;
    }

    uint items[2 * RESTIR_PT_MAX_NEIGHBORS];
    uint count = 0;
    float confidence_sum = canonical.M;
    for (uint s = 0; s < RESTIR_PT_MAX_NEIGHBORS; s++) {
        bool valid = active && s < landed;
        int2 prev = int2(-1);
        if (valid) {
            prev = restir_pt_unpack_pixel(pc.neighbors[restir_pt_slot(index, s)]);
            valid = restir_pt_in_screen(prev, dim);
        }
        if (valid && !merian_restir_pt_area)
            valid = params.prev_gbuffer.get_hit(uint2(prev)).is_valid() &&
                    params.gbuffer.get_hit(pixel).is_valid();
        if (active && !valid)
            pc.neighbors[restir_pt_slot(index, s)] = RESTIR_PT_NO_NEIGHBOR;
        if (!valid)
            continue;
        const uint prev_index = restir_pt_index(uint2(prev), dim);
        const RestirPTReservoir history = pc.reservoirs_prev[prev_index];
        confidence_sum += min(history.M, restir_pt_history_cap(prev_index));
        if (history.W > 0.f)
            items[count++] = restir_pt_item(index, s, 0);
        if (canonical.W > 0.f && !(merian_restir_pt_area && !backproject))
            items[count++] = restir_pt_item(index, s, 1);
    }
    // with area reservoirs the canonical path is shifted once, to where it splats
    if (merian_restir_pt_area && !backproject && active && canonical.W > 0.f &&
        pc.temporal_domains[4 * index] != RESTIR_PT_NO_NEIGHBOR)
        items[count++] = restir_pt_item(index, 0, 1);
    if (active)
        pc.confidence_sums[index] = confidence_sum;
    restir_pt_enqueue(params, items, count);
}

// Selects every pixel's spatial neighbors, sums the confidence of the ones it accepts, and queues
// the shifts they need. A reciprocal pair shares its shifts: each side computes the one into its
// own domain.
void restir_pt_pass_spatial_neighbors(const uint2 thread_pixel,
                                      const ParameterBlock<Scene> scene,
                                      const ParameterBlock<RestirPTBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    const uint2 pixel = min(thread_pixel, dim - 1);
    const bool active = all(thread_pixel < dim);
    const uint index = restir_pt_index(pixel, dim);
    const uint neighbor_count = uint(merian_restir_pt_neighbor_count);
    const bool reciprocal = merian_restir_pt_neighbor_selection == RestirPTNeighborsReciprocal;
    RandomGenerator rng = restir_pt_rng(pixel, RestirPTPassSpatialNeighbors);

    const RestirPTReservoir central = pc.reservoirs_in[index];
    const RestirPTSurface surface = restir_pt_surface(params.gbuffer, pixel);

    int2 candidates[RESTIR_PT_MAX_NEIGHBORS];
    if (merian_restir_pt_neighbor_selection == RestirPTNeighborsCGNS) {
        const RestirPTCgnsSelection selection =
            restir_pt_cgns_select(scene.camera, params.gbuffer, pixel, dim, pc.spatial_radius,
                                  pc.scale_solid_angle, pc.normal_beta, pc.early_stop_cutoff, rng);
        candidates = selection.selected;
    } else {
        const uint start = uint(rng.next_float() * RESTIR_PT_OFFSET_COUNT);
        for (uint s = 0; s < neighbor_count; s++) {
            candidates[s] =
                reciprocal
                    ? restir_pt_reciprocal_neighbor(pc.pairing, pc.pairing_transform, pixel, s)
                    : int2(round(float2(pixel) +
                                 pc.spatial_radius *
                                     restir_pt_disk_offset((start + s) % RESTIR_PT_OFFSET_COUNT)));
        }
    }

    uint items[2 * RESTIR_PT_MAX_NEIGHBORS];
    uint count = 0;
    float confidence_sum = central.M;
    const bool reuses = active && (merian_restir_pt_area || surface.is_surface());
    for (uint s = 0; s < neighbor_count; s++) {
        const int2 q = candidates[s];
        bool valid = reuses && restir_pt_in_screen(q, dim) && any(q != int2(pixel));
        if (valid)
            valid = restir_pt_compatible(surface, restir_pt_surface(params.gbuffer, uint2(q))) ||
                    (merian_restir_pt_area && !merian_restir_pt_geometry_rejection);
        if (active)
            pc.neighbors[restir_pt_slot(index, s)] =
                valid ? restir_pt_pack_pixel(q) : RESTIR_PT_NO_NEIGHBOR;
        if (!valid)
            continue;
        const RestirPTReservoir neighbor = pc.reservoirs_in[restir_pt_index(uint2(q), dim)];
        confidence_sum += neighbor.M;
        if (neighbor.W > 0.f)
            items[count++] = restir_pt_item(index, s, 0);
        if (!reciprocal && central.W > 0.f)
            items[count++] = restir_pt_item(index, s, 1);
    }
    if (active)
        pc.confidence_sums[index] = confidence_sum;
    restir_pt_enqueue(params, items, count);
}

// One queued shift: direction 0 moves the neighbor's path into this pixel's domain, direction 1
// this pixel's path into the neighbor's.
void restir_pt_pass_shift(const uint item_index,
                          const ParameterBlock<Scene> scene,
                          const ParameterBlock<RestirPTBinding> params) {
    if (item_index >= params.queue[3])
        return;
    const uint item = params.queue[4 + item_index];
    const uint index = item >> 4;
    const uint slot = (item >> 1) & 7u;
    const uint direction = item & 1u;

    const uint2 dim = params.gbuffer.get_dimensions();
    const uint2 pixel = uint2(index % dim.x, index / dim.x);
    RandomGenerator rng = restir_pt_rng(pixel, RestirPTPassShift);

    // what moves where: every branch ends in the one shift below, which is too large to inline
    // more than once
    const bool temporal = (pc.flags & RestirPTPassTemporalShifts) != 0;
    const bool coverage = (pc.flags & RestirPTPassCoverageShifts) != 0;
    const bool backprojected = temporal && restir_pt_subpixel_backprojection();
    const bool splat = temporal && restir_pt_subpixel_splatting();
    const uint2 neighbor = uint2(restir_pt_unpack_pixel(pc.neighbors[restir_pt_slot(index, slot)]));
    RestirPTReservoir r;
    float2 image_pos;
    bool prev_frame;
    uint result;
    if (coverage) {
        // a previous reservoir into a neighbor, within the previous frame
        r = pc.reservoirs_prev[index];
        restir_pt_from_prev_frame(scene, r);
        const int2 offset = RESTIR_PT_COVERAGE_OFFSETS[slot];
        image_pos = restir_pt_image_pos(float2(int2(pixel) + offset), r);
        prev_frame = true;
        result = restir_pt_coverage_shift_index(index, offset);
    } else if (direction == 0) {
        const uint src = restir_pt_index(neighbor, dim);
        r = backprojected ? pc.backprojected[index]
            : temporal    ? pc.reservoirs_prev[src]
                          : pc.reservoirs_in[src];
        if (temporal && !backprojected)
            restir_pt_from_prev_frame(scene, r);
        if (temporal && merian_restir_pt_dynamic_scene) {
            r = restir_pt_update_suffix(scene, r, false, rng);
            RestirPTSuffix suffix;
            suffix.rc_wo = r.rc_wo;
            suffix.light_pdf = r.light_pdf;
            suffix.rc_radiance = r.rc_radiance;
            suffix.unused = 0.f;
            pc.history[restir_pt_slot(index, slot)] = suffix;
        }
        image_pos = splat ? float2(pixel) + restir_pt_unpack_unorm(
                                                pc.splat_samples[2 * restir_pt_slot(index, slot)])
                          : restir_pt_image_pos(float2(pixel), r);
        prev_frame = false;
        result = restir_pt_shift_index(index, slot, 0);
    } else {
        // the inverse of the temporal update: what arrived at x_k in the previous frame
        r = pc.reservoirs_in[index];
        if (temporal && merian_restir_pt_dynamic_scene)
            r = restir_pt_update_suffix(scene, r, true, rng);
        const uint2 domain =
            uint2(pc.temporal_domains[4 * index], pc.temporal_domains[4 * index + 1]);
        image_pos =
            splat
                ? float2(restir_pt_unpack_pixel(domain.x)) + restir_pt_unpack_unorm(domain.y)
                : restir_pt_image_pos(
                      backprojected ? restir_pt_backprojected_origin(index) : float2(neighbor), r);
        prev_frame = temporal;
        result = restir_pt_shift_index(index, slot, 1);
    }
    const RestirPTPrimary primary =
        prev_frame ? restir_pt_primary(scene, params.prev_gbuffer, true, image_pos, dim, r.seed)
                   : restir_pt_primary(scene, params.gbuffer, false, image_pos, dim, r.seed);
    pc.shifts[result] = restir_pt_shift(scene, primary, r, prev_frame, rng);
}

// Streaming resampling of shifted reservoirs [Lin et al. 2022, eq. 19], w = m_i p_hat(T(x)) W |T'|.
struct RestirPTResampler {
    RestirPTReservoir r;
    float weight_sum;
    float3 decoupled;
    // shifts that produced a path with a non-zero target
    uint shifted;
    uint selected;
    uint added;

    __init() {
        r = restir_pt_empty_reservoir();
        weight_sum = 0.f;
        decoupled = float3(0);
        shifted = 0;
        selected = 0;
        added = 0;
    }

    // `film_jacobian` is the part of |T'| a splat adds on the film, which x_k's geometry term does
    // not carry.
    [mutating] void add(const RestirPTReservoir src,
                        const RestirPTShift shift,
                        const float mis,
                        inout RandomGenerator rng,
                        const float film_jacobian = 1.f) {
        r.M += src.M;
        added++;
        const float jacobian = shift.jacobian * film_jacobian;
        const float weight = yuv_luminance(shift.integrand) * jacobian * src.W * mis;
        if (!(weight > 0.f) || !isfinite(weight))
            return;
        shifted++;
        decoupled += shift.integrand * (jacobian * src.W * mis);
        weight_sum += weight;
        if (rng.next_float() * weight_sum < weight) {
            selected = added;
            const float M = r.M;
            r = src;
            r.M = M;
            r.integrand = shift.integrand;
            r.rc_geometry *= shift.jacobian;
        }
    }

    RestirPTReservoir finalize() {
        RestirPTReservoir out = r;
        const float target = yuv_luminance(out.integrand);
        out.W = target > 0.f ? weight_sum / target : 0.f;
        return out;
    }
}

RestirPTShift
restir_pt_identity(const RestirPTReservoir r) {
    RestirPTShift s;
    s.integrand = r.integrand;
    s.jacobian = 1.f;
    return s;
}

float restir_pt_shifted_target(const RestirPTShift s) {
    return yuv_luminance(s.integrand) * s.jacobian;
}

// The canonical path against the neighbors with the defensive pairwise MIS weights of
// [Lin et al. 2022, eq. 38], weighted by confidence [Hedstrom et al. 2026, eq. 11]. The neighbors
// are the previous frame's reservoirs where `temporal`.
RestirPTResampler restir_pt_resample_pairwise(const ParameterBlock<Scene> scene,
                                              const uint index,
                                              const uint2 dim,
                                              const RestirPTReservoir canonical,
                                              const bool temporal,
                                              inout RandomGenerator rng) {
    const bool reciprocal =
        !temporal && merian_restir_pt_neighbor_selection == RestirPTNeighborsReciprocal;
    const uint slots = temporal ? RESTIR_PT_MAX_NEIGHBORS : uint(merian_restir_pt_neighbor_count);
    const float confidence_sum = pc.confidence_sums[index];
    const float others = confidence_sum - canonical.M;
    const float target = yuv_luminance(canonical.integrand);

    RestirPTResampler resampler = RestirPTResampler();
    // the term that keeps the canonical sample defensive, eq. 11b
    float canonical_mis = canonical.M / confidence_sum;
    for (uint s = 0; s < slots; s++) {
        const uint packed = pc.neighbors[restir_pt_slot(index, s)];
        if (packed == RESTIR_PT_NO_NEIGHBOR)
            continue;
        const uint neighbor_index = restir_pt_index(uint2(restir_pt_unpack_pixel(packed)), dim);
        const RestirPTReservoir neighbor = temporal
                                               ? restir_pt_history(scene, neighbor_index, index, s)
                                               : pc.reservoirs_in[neighbor_index];

        // what the canonical path is worth in the neighbor's domain; a reciprocal neighbor
        // computed that shift for its own resampling
        const float canonical_there =
            canonical.W > 0.f
                ? restir_pt_shifted_target(
                      pc.shifts[reciprocal ? restir_pt_shift_index(neighbor_index, s, 0)
                                           : restir_pt_shift_index(index, s, 1)])
                : 0.f;
        const float canonical_pair = canonical_there * others + target * canonical.M;
        canonical_mis += neighbor.M / confidence_sum *
                         (canonical_pair > 0.f ? target * canonical.M / canonical_pair : 1.f);

        // and the neighbor's path here, eq. 11a
        if (neighbor.W > 0.f) {
            const RestirPTShift shift = pc.shifts[restir_pt_shift_index(index, s, 0)];
            const float neighbor_term = yuv_luminance(neighbor.integrand) * others;
            const float mis = neighbor.M / confidence_sum * neighbor_term /
                              (neighbor_term + restir_pt_shifted_target(shift) * canonical.M);
            resampler.add(neighbor, shift, isfinite(mis) ? mis : 0.f, rng);
        } else {
            resampler.r.M += neighbor.M;
        }
    }
    // the canonical sample needs the sum over the neighbors, so it goes in last
    resampler.add(canonical, restir_pt_identity(canonical), canonical_mis, rng);
    return resampler;
}

// The canonical path against the one the previous frame left at the backprojected pixel, with the
// confidence-weighted balance heuristic [Lin et al. 2022, eq. 36 and 6.2]. Splatted reservoirs
// are any number, weighed pairwise like spatial neighbors.
void restir_pt_pass_temporal(const uint2 pixel,
                             const ParameterBlock<Scene> scene,
                             const ParameterBlock<RestirPTBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;
    const uint index = restir_pt_index(pixel, dim);
    RandomGenerator rng = restir_pt_rng(pixel, RestirPTPassTemporal);
    const RestirPTReservoir canonical = pc.reservoirs_in[index];

    RestirPTResampler resampler = RestirPTResampler();
    if (restir_pt_subpixel_splatting()) {
        // every landed history against the canonical path, and the canonical path against the
        // domain it splats back to [Liu et al. 2025]
        const uint landed = min(pc.splat_counts[index], RESTIR_PT_MAX_NEIGHBORS);
        const float target = yuv_luminance(canonical.integrand);
        for (uint s = 0; s < landed; s++) {
            const uint packed = pc.neighbors[restir_pt_slot(index, s)];
            if (packed == RESTIR_PT_NO_NEIGHBOR)
                continue;
            RestirPTReservoir history = restir_pt_history(
                scene, restir_pt_index(uint2(restir_pt_unpack_pixel(packed)), dim), index, s);
            history.film = pc.splat_samples[2 * restir_pt_slot(index, s)];
            const float film_jacobian = asfloat(pc.splat_samples[2 * restir_pt_slot(index, s) + 1]);
            const RestirPTShift shift = pc.shifts[restir_pt_shift_index(index, s, 0)];
            const float history_term = yuv_luminance(history.integrand) * history.M;
            const float mis = history_term / (history_term + restir_pt_shifted_target(shift) *
                                                                 film_jacobian * canonical.M);
            resampler.add(history, shift, isfinite(mis) ? mis : 0.f, rng, film_jacobian);
        }
        float canonical_mis = 1.f;
        const float domain_confidence = asfloat(pc.temporal_domains[4 * index + 3]);
        if (canonical.W > 0.f && pc.temporal_domains[4 * index] != RESTIR_PT_NO_NEIGHBOR) {
            const float there =
                restir_pt_shifted_target(pc.shifts[restir_pt_shift_index(index, 0, 1)]) *
                asfloat(pc.temporal_domains[4 * index + 2]) * domain_confidence;
            if (there > 0.f)
                canonical_mis = canonical.M * target / (there + canonical.M * target);
        }
        resampler.add(canonical, restir_pt_identity(canonical), canonical_mis, rng);
        const int2 nearest =
            clamp(int2(floor(float2(pixel) + params.gbuffer.get_motion_vectors(pixel) + 0.5f)),
                  int2(0), int2(dim) - 1);
        const uint nearest_index = restir_pt_index(uint2(nearest), dim);
        resampler.r.M = canonical.M + min(pc.reservoirs_prev[nearest_index].M,
                                          restir_pt_history_cap(nearest_index));
    } else if (restir_pt_splatting()) {
        resampler = restir_pt_resample_pairwise(scene, index, dim, canonical, true, rng);
    } else {
        const uint packed = pc.neighbors[restir_pt_slot(index, 0)];
        float canonical_mis = 1.f;
        if (packed != RESTIR_PT_NO_NEIGHBOR) {
            const RestirPTReservoir history = restir_pt_history(
                scene, restir_pt_index(uint2(restir_pt_unpack_pixel(packed)), dim), index, 0);
            const float target = yuv_luminance(canonical.integrand);
            const float canonical_there =
                canonical.W > 0.f
                    ? restir_pt_shifted_target(pc.shifts[restir_pt_shift_index(index, 0, 1)])
                    : 0.f;
            if (canonical_there > 0.f)
                canonical_mis =
                    canonical.M * target / (canonical_there * history.M + canonical.M * target);
            if (history.W > 0.f) {
                const RestirPTShift shift = pc.shifts[restir_pt_shift_index(index, 0, 0)];
                const float history_term = yuv_luminance(history.integrand) * history.M;
                const float mis =
                    history_term / (history_term + restir_pt_shifted_target(shift) * canonical.M);
                resampler.add(history, shift, isfinite(mis) ? mis : 0.f, rng);
            } else {
                resampler.r.M += history.M;
            }
        }
        resampler.add(canonical, restir_pt_identity(canonical), canonical_mis, rng);
    }

    const RestirPTReservoir out = resampler.finalize();
    pc.reservoirs_out[index] = out;
    if ((pc.flags & RestirPTPassResolves) != 0)
        restir_pt_resolve(params, pixel, out, resampler.decoupled);
    restir_pt_debug_reuse(params, pixel, resampler.added - 1, resampler.added - 1,
                          resampler.shifted - (canonical.W > 0.f ? 1 : 0),
                          resampler.selected != resampler.added);
}

// Resamples the neighbors the spatial neighbor pass selected.
void restir_pt_pass_spatial(const uint2 pixel,
                            const ParameterBlock<Scene> scene,
                            const ParameterBlock<RestirPTBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;
    const uint index = restir_pt_index(pixel, dim);
    RandomGenerator rng = restir_pt_rng(pixel, RestirPTPassSpatial);
    const RestirPTReservoir canonical = pc.reservoirs_in[index];
    const RestirPTResampler resampler =
        restir_pt_resample_pairwise(scene, index, dim, canonical, false, rng);

    const RestirPTReservoir out = resampler.finalize();
    pc.reservoirs_out[index] = out;
    if ((pc.flags & RestirPTPassResolves) != 0)
        restir_pt_resolve(params, pixel, out, resampler.decoupled);
    restir_pt_debug_reuse(params, pixel, uint(merian_restir_pt_neighbor_count), resampler.added - 1,
                          resampler.shifted - (canonical.W > 0.f ? 1 : 0),
                          resampler.selected != resampler.added);
}

static const uint RESTIR_PT_TILE = 8 + 2 * RESTIR_PT_DUPLICATION_RADIUS;
groupshared uint restir_pt_tile_seeds[RESTIR_PT_TILE][RESTIR_PT_TILE];

// How many pixels of the 17x17 window around each pixel hold the same path, and for how many
// frames the pixel has kept its path. Read by the next frame's history cap.
void restir_pt_pass_duplication(const uint2 pixel,
                                const uint2 group_pixel,
                                const uint2 group,
                                const uint group_index,
                                const ParameterBlock<RestirPTBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    const int2 corner = int2(group * 8) - int(RESTIR_PT_DUPLICATION_RADIUS);
    for (uint t = group_index; t < RESTIR_PT_TILE * RESTIR_PT_TILE; t += 64) {
        const uint2 local = uint2(t % RESTIR_PT_TILE, t / RESTIR_PT_TILE);
        const int2 p = corner + int2(local);
        uint seed = 0;
        if (restir_pt_in_screen(p, dim)) {
            const RestirPTReservoir r = pc.reservoirs_in[restir_pt_index(uint2(p), dim)];
            seed = r.W > 0.f ? r.seed : 0u;
        }
        restir_pt_tile_seeds[local.y][local.x] = seed;
    }
    GroupMemoryBarrierWithGroupSync();
    if (any(pixel >= dim))
        return;

    const uint2 center = group_pixel + RESTIR_PT_DUPLICATION_RADIUS;
    const uint own = restir_pt_tile_seeds[center.y][center.x];
    uint count = 0;
    if (own != 0) {
        for (int j = -int(RESTIR_PT_DUPLICATION_RADIUS); j <= int(RESTIR_PT_DUPLICATION_RADIUS);
             j++) {
            for (int i = -int(RESTIR_PT_DUPLICATION_RADIUS); i <= int(RESTIR_PT_DUPLICATION_RADIUS);
                 i++) {
                if (restir_pt_tile_seeds[int(center.y) + j][int(center.x) + i] == own)
                    count++;
            }
        }
        count--;
    }

    const uint index = restir_pt_index(pixel, dim);
    const int2 prev = int2(round(float2(pixel) + params.gbuffer.get_motion_vectors(pixel)));
    uint age = 0;
    if (own != 0 && restir_pt_in_screen(prev, dim)) {
        const uint prev_index = restir_pt_index(uint2(prev), dim);
        if (pc.reservoirs_prev[prev_index].seed == own)
            age = min((pc.prev_duplication[prev_index] >> 16) + 1, 0xFFFFu);
    }
    pc.duplication[index] = count | (age << 16);

    if (merian_restir_pt_debug_view == RestirPTDebugDuplication) {
        const float3 view =
            float3(float(count) / RESTIR_PT_DUPLICATION_WINDOW, min(float(age) / 40.f, 1.f), 0);
        params.irradiance[pixel] = float4(view, 0);
    }
}

// One pipeline per resampling pass, specialized on the pass.
[vk::constant_id(0)] const uint restir_pt_pass = RestirPTPassTemporalNeighbors;

[shader("compute")][numthreads(8, 8, 1)] void resample(uint3 thread_id : SV_DispatchThreadID,
                                                       uint3 group_thread : SV_GroupThreadID,
                                                       uint3 group : SV_GroupID,
                                                       uint group_index : SV_GroupIndex,
                                                       ParameterBlock<Scene> scene,
                                                       ParameterBlock<RestirPTBinding> params) {
    switch (restir_pt_pass) {
    case RestirPTPassTemporalNeighbors:
        restir_pt_pass_temporal_neighbors(thread_id.xy, scene, params);
        break;
    case RestirPTPassSpatialNeighbors:
        restir_pt_pass_spatial_neighbors(thread_id.xy, scene, params);
        break;
    case RestirPTPassTemporal:
        restir_pt_pass_temporal(thread_id.xy, scene, params);
        break;
    case RestirPTPassSpatial:
        restir_pt_pass_spatial(thread_id.xy, scene, params);
        break;
    case RestirPTPassDuplication:
        restir_pt_pass_duplication(thread_id.xy, group_thread.xy, group.xy, group_index, params);
        break;
    case RestirPTPassSplat:
        restir_pt_pass_splat(thread_id.xy, scene, params);
        break;
    case RestirPTPassSubpixelBackprojection:
        restir_pt_pass_subpixel_backprojection(thread_id.xy, scene, params);
        break;
    case RestirPTPassCoverageNeighbors:
        restir_pt_pass_coverage_neighbors(thread_id.xy, params);
        break;
    case RestirPTPassRecord:
        restir_pt_pass_record(thread_id.xy, scene, params);
        break;
    default:
        break;
    }
}

    [shader("compute")][numthreads(8, 8, 1)] void initial(uint3 thread_id : SV_DispatchThreadID,
                                                          ParameterBlock<Scene> scene,
                                                          ParameterBlock<RestirPTBinding> params) {
    restir_pt_pass_initial(thread_id.xy, scene, params);
}

[shader("raygeneration")] void initial_rt(ParameterBlock<Scene> scene,
                                          ParameterBlock<RestirPTBinding> params) {
    restir_pt_pass_initial(DispatchRaysIndex().xy, scene, params);
}

    [shader("compute")][numthreads(64, 1, 1)] void shift(uint3 thread_id : SV_DispatchThreadID,
                                                         ParameterBlock<Scene> scene,
                                                         ParameterBlock<RestirPTBinding> params) {
    restir_pt_pass_shift(thread_id.x, scene, params);
}
