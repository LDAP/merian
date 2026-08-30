#include "merian-graph/nodes/render_restir_cgns/render_restir_cgns_neighbors.slangh"
#include "merian-graph/nodes/render_restir_cgns/render_restir_cgns_paths.slangh"

using merian;

namespace merian {
extern static const bool merian_cgns_demodulate_albedo = false;
extern static const bool merian_cgns_geometry_rejection = true;
extern static const bool merian_cgns_confidence_temporal = true;
extern static const bool merian_cgns_confidence_spatial = true;
// neighbors whose normals or depths diverge this far from the pixel's are not resampled
extern static const float merian_cgns_reject_normal = 0.5f;
extern static const float merian_cgns_reject_depth = 0.1f;
// how a previous reservoir reaches a pixel: pulled along the motion vector, pushed by projecting
// its own primary vertex, or pushed with the pull as a fallback where nothing lands
static const int CGNS_TEMPORAL_GATHER = 0;
static const int CGNS_TEMPORAL_SPLAT = 1;
static const int CGNS_TEMPORAL_SPLAT_GATHER = 2;
extern static const int merian_cgns_temporal_mode = CGNS_TEMPORAL_GATHER;
} // namespace merian

void cgns_write_irradiance(const CgnsBinding params,
                           const uint2 pixel,
                           const CgnsReservoir reservoir) {
    float3 irradiance = reservoir.integrand * reservoir.get_ucw();
    if (any(isnan(irradiance)) || any(isinf(irradiance)))
        irradiance = float3(0);

    if (merian_cgns_demodulate_albedo)
        irradiance /= params.gbuffer.get_demodulation_albedo(pixel);

    const float l = yuv_luminance(irradiance);
    params.irradiance[pixel] = float4(irradiance, l * l);
}

// One path per sample, resampled into a single reservoir. All samples are independent, so each
// enters with m_i = 1 / spp.
void cgns_pass_initial(const uint2 pixel,
                       const ParameterBlock<Scene> scene,
                       const ParameterBlock<CgnsBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    RandomGenerator rng = cgns_rng(pixel, 0);
    const CgnsVertex vertex =
        cgns_vertex(scene, params.gbuffer, scene.generate_camera_ray(pixel, dim), pixel);

    CgnsReservoir reservoir = CgnsReservoir();
    CgnsReconnection reconnection = CgnsReconnection();
    if (vertex.valid) {
        for (int s = 0; s < merian_cgns_spp; s++) {
            const CgnsCandidate candidate = cgns_trace_candidate(scene, vertex, pixel, rng);
            const float weight = candidate.weight_sum / float(merian_cgns_spp);
            reservoir.weight_sum += weight;
            if (rng.next_float() * reservoir.weight_sum < weight) {
                reservoir.integrand = candidate.integrand;
                reconnection = candidate.reconnection;
            }
        }
    }
    reservoir.confidence = 1.f;

    const uint index = cgns_index(pixel, dim);
    pc.reservoirs_out[index] = reservoir;
    pc.reconnection_out[index] = reconnection;
    if ((pc.flags & CgnsPassFlags::CgnsPassResolves) != 0)
        cgns_write_irradiance(params, pixel, reservoir);
}

// Splats a previous reservoir onto the pixel its own primary vertex projects to this frame
// [Liu et al. 2025], which reprojects the vertex itself instead of trusting a screen-space motion
// vector. A pixel more reservoirs land on than it has slots keeps the first of them.
void cgns_splat(const uint2 pixel,
                const ParameterBlock<Scene> scene,
                const ParameterBlock<CgnsBinding> params,
                const uint2 dim) {
    // Every reservoir with a domain of its own is splatted, empty ones included: the destination
    // weighs its canonical path against each of them, and one that carries nothing still competes.
    const uint index = cgns_index(pixel, dim);
    const CgnsVertex vertex = cgns_vertex(
        scene, params.prev_gbuffer, scene.prev_camera.generate_ray_pinhole(pixel, dim), pixel);
    // A primary vertex on the environment map sits far along its ray, so projecting it matches
    // directions rather than positions, which is what a distant vertex needs.
    if (!vertex.valid)
        return;
    if (!(dot(vertex.sp.pos - scene.camera.position, scene.camera.W) > 0.f))
        return;

    // Unjittered: a domain here is a whole pixel, so the sub-pixel offset the two frames were
    // sampled with says nothing about which of them the path belongs to.
    const float2 splat = scene.camera.project_pinhole(vertex.sp.pos, dim);
    if (!all(splat > float2(-0.5f)) || !all(splat < float2(dim) - 0.5f))
        return;

    const uint target = cgns_index(uint2(round(splat)), dim);
    uint slot;
    InterlockedAdd(params.splat_counts[target], 1u, slot);
    if (slot == 0)
        pc.splats[target] = index;
}

// Bilinear confidence over the footprint the pixel reprojects into, rather than the one reservoir
// it lands on, so it does not step as the reprojection crosses pixel borders [Liu et al. 2025].
float cgns_prev_confidence(const float2 prev_float, const uint2 dim) {
    const int2 top_left = int2(floor(prev_float));
    const float2 fractional = saturate(prev_float - float2(top_left));
    float confidence = 0.f;
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            const int2 neighbor = top_left + int2(x, y);
            if (any(neighbor < 0) || any(uint2(neighbor) >= dim))
                continue;
            const float bilinear =
                lerp(1.f - x, float(x), fractional.x) * lerp(1.f - y, float(y), fractional.y);
            confidence += bilinear * pc.reservoirs_prev[cgns_index(uint2(neighbor), dim)].confidence;
        }
    }
    return confidence;
}

// Resamples one previous-frame reservoir into this pixel, with the generalized Talbot MIS weight
// of [Lin et al. 2022, eq. 36] scaled by each reservoir's confidence [Lin et al. 2022, 6.2].
void cgns_resample_prev(const ParameterBlock<Scene> scene,
                        const CgnsVertex vertex,
                        const float current_confidence,
                        const bool prev_valid,
                        const CgnsReservoir prev,
                        CgnsReconnection prev_rec,
                        inout CgnsReservoir dst,
                        inout CgnsReconnection dst_rec,
                        inout RandomGenerator rng) {
    float mis = 0.f;
    float3 radiance = float3(0);
    float jacobian = 1.f;
    if (prev_valid && any(prev.integrand > 0.f)) {
        const CgnsShifted shifted = cgns_shift(scene, vertex, prev_rec, rng);
        const float shifted_jacobian =
            prev_rec.secondary_pdf > 0.f ? shifted.pdf / prev_rec.secondary_pdf : 0.f;
        const float m_here = yuv_luminance(shifted.radiance) * shifted_jacobian * current_confidence;
        if (isfinite(m_here) && m_here > 0.f) {
            radiance = shifted.radiance;
            jacobian = shifted_jacobian;
            // the path now lives in this domain
            prev_rec.secondary_pdf = shifted.pdf;
        }
        const float m_there = yuv_luminance(prev.integrand) * prev.confidence;
        mis = m_here + m_there > 0.f ? m_there / (m_here + m_there) : 0.f;
        mis = isfinite(mis) ? mis : 0.f;
    }
    if (dst.add(rng, mis, radiance, jacobian, prev))
        dst_rec = prev_rec;
}

// Resamples this pixel's canonical path against the ones the previous frame left for it, each
// shifted into the other's domain. The previous reservoirs come either from the pixel this one
// reprojects to along its motion vector, or from those that splatted themselves here.
//
// The splat dispatch shares this entry point: a sixth one using the scene block does not link.
void cgns_pass_temporal(const uint2 pixel,
                        const ParameterBlock<Scene> scene,
                        const ParameterBlock<CgnsBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    if ((pc.flags & CgnsPassFlags::CgnsPassSplats) != 0) {
        cgns_splat(pixel, scene, params, dim);
        return;
    }

    const uint index = cgns_index(pixel, dim);
    RandomGenerator rng = cgns_rng(pixel, 0);

    const CgnsReservoir current = pc.reservoirs_in[index];
    const CgnsReconnection current_rec = pc.reconnection_in[index];
    const CgnsVertex vertex =
        cgns_vertex(scene, params.gbuffer, scene.generate_camera_ray(pixel, dim), pixel);
    const float current_confidence = merian_cgns_confidence_temporal ? current.confidence : 1.f;

    // 1. the confidence a gathered reservoir carries. A splatted one names itself, so it brings
    // its own; either way the destination ends up with the history it actually resampled.
    const float2 prev_float = float2(pixel) + params.gbuffer.get_motion_vectors(pixel);
    const bool reprojects = all(prev_float > float2(-0.5f)) && all(prev_float < float2(dim) - 0.5f);
    const float inherited =
        !reprojects ? 0.f
                    : (merian_cgns_confidence_temporal ? cgns_prev_confidence(prev_float, dim)
                                                       : 1.f);

    // A pixel keeps one reservoir: the weights below pair it with the canonical path, and a
    // second one would be weighed against that same path and count itself in as well.
    const bool splatted =
        merian_cgns_temporal_mode != CGNS_TEMPORAL_GATHER && params.splat_counts[index] > 0;
    const bool gather =
        merian_cgns_temporal_mode == CGNS_TEMPORAL_GATHER ||
        (merian_cgns_temporal_mode == CGNS_TEMPORAL_SPLAT_GATHER && !splatted);

    CgnsReservoir dst = CgnsReservoir();
    CgnsReconnection dst_rec = CgnsReconnection();

    if (gather) {
        // one previous reservoir, so the pair of them carries the whole MIS weight
        const uint prev_index =
            cgns_index(uint2(clamp(int2(round(prev_float)), int2(0), int2(dim) - 1)), dim);
        CgnsVertex prev_vertex;
        prev_vertex.valid = false;
        CgnsReservoir prev = CgnsReservoir();
        CgnsReconnection prev_rec = CgnsReconnection();
        if (reprojects) {
            const uint2 prev_pixel = uint2(prev_index % dim.x, prev_index / dim.x);
            prev_vertex =
                cgns_vertex(scene, params.prev_gbuffer,
                            scene.prev_camera.generate_ray_pinhole(prev_pixel, dim), prev_pixel);
            prev = pc.reservoirs_prev[prev_index];
            prev.confidence = inherited;
            prev_rec = pc.reconnection_prev[prev_index];
        }

        float current_mis = 0.f;
        if (any(current.integrand > 0.f)) {
            const float m_here = yuv_luminance(current.integrand) * current_confidence;
            float m_there = 0.f;
            if (prev_vertex.valid) {
                const CgnsShifted shifted = cgns_shift(scene, prev_vertex, current_rec, rng);
                m_there = yuv_luminance(shifted.radiance) * prev.confidence *
                          (current_rec.secondary_pdf > 0.f ? shifted.pdf / current_rec.secondary_pdf
                                                           : 0.f);
                m_there = isfinite(m_there) ? m_there : 0.f;
            }
            current_mis = m_here + m_there > 0.f ? m_here / (m_here + m_there) : 0.f;
        }
        if (dst.add(rng, current_mis, current.integrand, 1.f, current))
            dst_rec = current_rec;

        cgns_resample_prev(scene, vertex, current_confidence, prev_vertex.valid, prev, prev_rec,
                           dst, dst_rec, rng);
    } else {
        // The splatted reservoir and this pixel's canonical path are the pair that competes
        // for it, weighed by the balance heuristic of [Lin et al. 2022, eq. 36]. The reference
        // pairs the canonical with the reservoir at the pixel it reprojects onto instead, which is
        // the one that splatted here only while the two maps invert each other; a pixel index is
        // too coarse for that, so the pairing follows the splat.
        const uint source = pc.splats[index];
        const uint2 source_pixel = uint2(source % dim.x, source / dim.x);
        const CgnsVertex source_vertex =
            cgns_vertex(scene, params.prev_gbuffer,
                        scene.prev_camera.generate_ray_pinhole(source_pixel, dim), source_pixel);
        const float source_confidence =
            merian_cgns_confidence_temporal ? pc.reservoirs_prev[source].confidence : 1.f;

        float current_mis = 0.f;
        if (any(current.integrand > 0.f)) {
            const float m_here = yuv_luminance(current.integrand) * current_confidence;
            float m_there = 0.f;
            if (source_vertex.valid) {
                const CgnsShifted shifted = cgns_shift(scene, source_vertex, current_rec, rng);
                m_there = yuv_luminance(shifted.radiance) * source_confidence *
                          (current_rec.secondary_pdf > 0.f ? shifted.pdf / current_rec.secondary_pdf
                                                           : 0.f);
                m_there = isfinite(m_there) ? m_there : 0.f;
            }
            current_mis = m_here + m_there > 0.f ? m_here / (m_here + m_there) : 0.f;
        }
        if (dst.add(rng, current_mis, current.integrand, 1.f, current))
            dst_rec = current_rec;

        CgnsReservoir prev = pc.reservoirs_prev[source];
        prev.confidence = source_confidence;
        cgns_resample_prev(scene, vertex, current_confidence, source_vertex.valid, prev,
                           pc.reconnection_prev[source], dst, dst_rec, rng);

        // the history a pixel inherits follows its motion vector, not what happened to land on it
        dst.confidence = min(current.confidence + inherited, merian_cgns_confidence_cap);
    }

    pc.reservoirs_out[index] = dst;
    pc.reconnection_out[index] = dst_rec;
    if ((pc.flags & CgnsPassFlags::CgnsPassResolves) != 0)
        cgns_write_irradiance(params, pixel, dst);
}

// Writes the neighbors every pixel resamples from.
void cgns_pass_select_neighbors(const uint2 pixel,
                                const ParameterBlock<Scene> scene,
                                const ParameterBlock<CgnsBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    RandomGenerator rng = cgns_rng(pixel, pc.spatial_round * 1337);
    const CgnsNeighbors neighbors =
        cgns_select_neighbors(scene, params.gbuffer, pixel, dim, pc.spatial_radius,
                              pc.scale_solid_angle, pc.normal_beta, pc.early_stop_cutoff, rng);

    const uint base = cgns_index(pixel, dim) * uint(merian_cgns_neighbor_count);
    for (uint i = 0; i < uint(merian_cgns_neighbor_count); i++)
        pc.neighbors[base + i] = neighbors.pixels[i];
}

// Resamples the pre-selected neighbors into this pixel with the defensive pairwise resampling MIS
// weights of [Lin et al. 2022, eq. 38], in the confidence-weighted form of
// [Hedstrom et al. 2026, eq. 11].
void cgns_pass_spatial(const uint2 pixel,
                       const ParameterBlock<Scene> scene,
                       const ParameterBlock<CgnsBinding> params) {
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    const uint index = cgns_index(pixel, dim);
    RandomGenerator rng = cgns_rng(pixel, pc.spatial_round * 1337);

    const CgnsVertex vertex =
        cgns_vertex(scene, params.gbuffer, scene.generate_camera_ray(pixel, dim), pixel);
    const SurfaceData central = params.gbuffer.get_surface(pixel);
    const bool central_surface = cgns_is_surface(central.linear_z);

    const CgnsReservoir central_res = pc.reservoirs_in[index];
    const CgnsReconnection central_rec = pc.reconnection_in[index];
    const float central_confidence = merian_cgns_confidence_spatial ? central_res.confidence : 1.f;
    const float central_target = yuv_luminance(central_res.integrand);

    // 1. which neighbors survive, and how much confidence they carry between them. The pairwise
    // weights need that sum (c in eq. 11) before the first pair is weighed, so acceptance cannot
    // be decided inside the loop below.
    int2 accepted_pixel[CGNS_MAX_NEIGHBOR_COUNT];
    float accepted_confidence[CGNS_MAX_NEIGHBOR_COUNT];
    int accepted = 0;
    float confidence_sum = 0.f;

    for (uint i = 0; i < uint(merian_cgns_neighbor_count); i++) {
        const int2 npixel = pc.neighbors[index * uint(merian_cgns_neighbor_count) + i];
        if (any(npixel < 0) || any(uint2(npixel) >= dim))
            continue;

        if (merian_cgns_geometry_rejection) {
            const SurfaceData ns = params.gbuffer.get_surface(uint2(npixel));
            if (central_surface != cgns_is_surface(ns.linear_z))
                continue;
            if (central_surface &&
                (dot(central.normal, ns.normal) < merian_cgns_reject_normal ||
                 abs(central.linear_z - ns.linear_z) >=
                     merian_cgns_reject_depth * central.linear_z))
                continue;
        }

        accepted_pixel[accepted] = npixel;
        accepted_confidence[accepted] =
            merian_cgns_confidence_spatial
                ? pc.reservoirs_in[cgns_index(uint2(npixel), dim)].confidence
                : 1.f;
        confidence_sum += accepted_confidence[accepted];
        accepted++;
    }

    const float total_confidence = confidence_sum + central_confidence;

    CgnsReservoir dst = CgnsReservoir();
    CgnsReconnection dst_rec = CgnsReconnection();
    // the term that keeps the canonical sample defensive, eq. 11b
    float central_mis = total_confidence > 0.f ? central_confidence / total_confidence : 1.f;

    for (int i = 0; i < accepted; i++) {
        const uint2 npixel = uint2(accepted_pixel[i]);
        const float nconfidence = accepted_confidence[i];
        const uint nindex = cgns_index(npixel, dim);
        const CgnsReservoir nres = pc.reservoirs_in[nindex];
        CgnsReconnection nrec = pc.reconnection_in[nindex];

        // What this pixel's own path is worth in the neighbor's domain. Shading the neighbor is
        // the expensive half of the pair, so nothing loads it for a pixel that has no path.
        float target_there = 0.f;
        if (central_target > 0.f) {
            const CgnsVertex nvertex = cgns_vertex(
                scene, params.gbuffer, scene.generate_camera_ray(npixel, dim), npixel);
            const CgnsShifted shifted = cgns_shift(scene, nvertex, central_rec, rng);
            target_there =
                yuv_luminance(shifted.radiance) *
                (central_rec.secondary_pdf > 0.f ? shifted.pdf / central_rec.secondary_pdf : 0.f);
            target_there = isfinite(target_there) ? target_there : 0.f;
        }
        const float central_pair =
            confidence_sum * target_there + central_confidence * central_target;
        if (central_pair > 0.f) {
            central_mis += (nconfidence / total_confidence) *
                           (central_confidence * central_target) / central_pair;
        }

        // and the neighbor's path here, eq. 11a
        float nmis = 0.f;
        float3 nradiance = float3(0);
        float njacobian = 1.f;
        if (any(nres.integrand > 0.f)) {
            const CgnsShifted shifted = cgns_shift(scene, vertex, nrec, rng);
            const float jacobian =
                nrec.secondary_pdf > 0.f ? shifted.pdf / nrec.secondary_pdf : 0.f;
            float target_here = yuv_luminance(shifted.radiance) * jacobian;
            if (isfinite(target_here) && target_here > 0.f) {
                nradiance = shifted.radiance;
                njacobian = jacobian;
                // the path now lives in this domain
                nrec.secondary_pdf = shifted.pdf;
            } else {
                target_here = 0.f;
            }
            const float ntarget = yuv_luminance(nres.integrand);
            const float pair = confidence_sum * ntarget + central_confidence * target_here;
            if (pair > 0.f) {
                nmis = (confidence_sum / total_confidence) * (nconfidence * ntarget) / pair;
            }
        }
        if (dst.add(rng, nmis, nradiance, njacobian, nres))
            dst_rec = nrec;
    }

    // the canonical sample needed the sum over the neighbors, so it goes in last
    if (dst.add(rng, central_mis, central_res.integrand, 1.f, central_res))
        dst_rec = central_rec;

    pc.reservoirs_out[index] = dst;
    pc.reconnection_out[index] = dst_rec;
    if ((pc.flags & CgnsPassFlags::CgnsPassResolves) != 0)
        cgns_write_irradiance(params, pixel, dst);
}

// The candidate trace also runs from a raygen shader: the ray tracing pipeline schedules its long,
// divergent paths better than a compute dispatch does.
[shader("raygeneration")] void initial_rt(ParameterBlock<Scene> scene,
                                          ParameterBlock<CgnsBinding> params) {
    cgns_pass_initial(DispatchRaysIndex().xy, scene, params);
}

[shader("compute")][numthreads(8, 8, 1)] void initial(uint3 thread_id: SV_DispatchThreadID,
                                                      ParameterBlock<Scene> scene,
                                                      ParameterBlock<CgnsBinding> params) {
    cgns_pass_initial(thread_id.xy, scene, params);
}

[shader("compute")][numthreads(8, 8, 1)] void temporal(uint3 thread_id: SV_DispatchThreadID,
                                                       ParameterBlock<Scene> scene,
                                                       ParameterBlock<CgnsBinding> params) {
    cgns_pass_temporal(thread_id.xy, scene, params);
}

[shader("compute")][numthreads(8, 8, 1)] void select_neighbors(uint3 thread_id: SV_DispatchThreadID,
                                                               ParameterBlock<Scene> scene,
                                                               ParameterBlock<CgnsBinding> params) {
    cgns_pass_select_neighbors(thread_id.xy, scene, params);
}

[shader("compute")][numthreads(8, 8, 1)] void spatial(uint3 thread_id: SV_DispatchThreadID,
                                                      ParameterBlock<Scene> scene,
                                                      ParameterBlock<CgnsBinding> params) {
    cgns_pass_spatial(thread_id.xy, scene, params);
}
