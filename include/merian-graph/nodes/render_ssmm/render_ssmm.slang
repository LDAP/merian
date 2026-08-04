#include "merian-graph/nodes/render_ssmm/render_ssmm.slangh"

using merian;

namespace merian {
extern static const int merian_ssmm_spp = 1;
extern static const float merian_ssmm_bsdf_p = 0.15;
extern static const float merian_ssmm_ml_prior_n = 0.2;
extern static const uint merian_ssmm_ml_max_n = 1024u;
extern static const float merian_ssmm_ml_min_alpha = 0.01;
extern static const int merian_ssmm_smis_group_size = 5;
}

static const int SSMM_MAX_SPP = 16;

struct SSMMBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> irradiance;
};

typealias mc_state_t = SSMCState;

// Ported from VKDT: https://github.com/hanatos/vkdt
// with approval from author.

bool mc_state_valid(mc_state_t s) { return s.sum_w > 0.0; }

float3 mc_state_dir(mc_state_t s, float3 x) {
    float3 tgt = s.sum_tgt / (s.sum_w > 0.0 ? s.sum_w : 1.0);
    return normalize(tgt - x);
}

// add sample to lobe via maximum likelihood estimator and exponentially weighted average
void mc_state_add(inout mc_state_t s, float3 x, float w, float3 dir, float3 y) {
    s.N = min(s.N + 1, merian_ssmm_ml_max_n);
    // float alpha = 0.999; // for faster adaptation!
    float alpha = max(1.0 / s.N, merian_ssmm_ml_min_alpha);
    s.sum_w   = lerp(s.sum_w, w, alpha);
    s.sum_tgt = lerp(s.sum_tgt, w * y, alpha);
    float3 to = s.sum_len * mc_state_dir(s, x);
    to = lerp(to, w * dir, alpha);
    s.sum_len = length(to);
}

// return a vmf lobe (direction + kappa) for the given position
float4 mc_state_vmf(mc_state_t s, float3 x) {
    float r = s.sum_len / (s.sum_w > 0.0 ? s.sum_w : 1.0); // = mean cosine in [0,1]
    r = (s.N * s.N * r) / (s.N * s.N + merian_ssmm_ml_prior_n);
    return float4(mc_state_dir(s, x), (3.0 * r - r * r * r) / (1.0 - r * r));
}

// shuffle state to different subgroup lane
void mc_state_shuffle(inout mc_state_t s, uint lane) {
    s.N       = WaveShuffle(s.N, lane);
    s.sum_w   = WaveShuffle(s.sum_w, lane);
    s.sum_len = WaveShuffle(s.sum_len, lane);
    s.sum_tgt = WaveShuffle(s.sum_tgt, lane);
    s.f       = WaveShuffle(s.f, lane);
}

// zero-init new markov chain state
mc_state_t mc_state_new() {
    mc_state_t r = { float3(0.0), 0.0, 0, 0.0, 0.0 };
    return r;
}

// evaluate hemisphere clamp test (can this light y be seen from x with normal nx?)
float mc_state_C(mc_state_t s,
                 GBuffer gbuffer,
                 float3 cam_x,
                 uint2 resolution,
                 float3 x,
                 float3 nx,
                 int2 tc,
                 int2 tc0) {
    // return 1.0;
    const uint2 q = uint2(clamp(tc, int2(0), int2(resolution) - 1));
    // const float irr0 = gbuf.w;
    // const float sig0 = sqrt(max(0, gbuf.w - gbuf.z*gbuf.z)); // noise stddev
    // const float sig1 = sqrt(max(0, gbuf.w - gbuf.z*gbuf.z)); // noise stddev
    float3 nq = gbuffer.get_normal(q);
    const float sigma_n = 64; // svgf style
    const float sigma_z = 10;
    const float sigma_r = 100; // lower to like 0.1 for sharp static contact shadows.
                               // unfortunately dynamic/rotating lights require a looser bound here
                               // also estimating variance is expensive (texture access), so disabled for now:
    // float w_r = exp(-abs(gbuf.w - irr0) / (sigma_r*(sig0+sig1)));
    float w_n = pow(max(0.0, dot(nx, nq)), sigma_n);
    float w_d = exp(-abs(gbuffer.get_linear_z(q) - length(x - cam_x)) / sigma_z);
    return w_n * w_d; // * w_r;
}

float mc_state_score(mc_state_t s,
                     GBuffer gbuffer,
                     float3 cam_x,
                     uint2 resolution,
                     float3 x,
                     float3 nx,
                     float rnd0,
                     int2 tc,
                     int2 tc0) {
    const float f = s.f;
    return f * mc_state_C(s, gbuffer, cam_x, resolution, x, nx, tc, tc0);
}

void read_neighbour_state(GBuffer gbuffer,
                          float3 cam_x,
                          uint2 resolution,
                          inout mc_state_t tent,
                          float3 x,
                          float3 n,
                          int2 ipos,
                          float2 mv,
                          inout RandomGenerator rng) {
    float sum = mc_state_score(tent, gbuffer, cam_x, resolution, x, n, rng.next_float(),
                               int2(float2(ipos) + mv), int2(float2(ipos) + mv));
    for (int i = 0; i < merian_ssmm_smis_group_size; i++) {
        // helps explore difficult cases but is expensive
        float4 r0 = rng.next_float4();
        float4 r1 = rng.next_float4();
        float4 r2 = rng.next_float4();
        int2 off = int2(15.0 * (r0.xy + r0.zw + r1.xy + r1.zw + r2.xy + r2.zw - 3.0));
        float rnd = rng.next_float();
        mc_state_t t2 = pc.ssmc_prev[ssmm_index(
            clamp(int2(float2(ipos) + mv) + off, int2(0), int2(resolution) - 1), resolution)];
        float other = mc_state_score(t2, gbuffer, cam_x, resolution, x, n, rng.next_float(),
                                     int2(float2(ipos) + mv) + off, int2(float2(ipos) + mv));
        if (sum <= 0.0 || rnd < other / (other + sum))
            tent = t2;
        sum += other;
    }
}

[shader("raygeneration")]
void main(ParameterBlock<Scene> scene, ParameterBlock<SSMMBinding> params) {
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 resolution = params.gbuffer.get_dimensions();
    if (any(pixel >= resolution))
        return;

    RandomGenerator rng = XorShift32(pcg4d16(uint4(pixel, pc.frame, pc.seed)));
    float2 moments = float2(0);
    float3 irr = float3(0);

    const float2 mv = params.gbuffer.get_motion_vectors(pixel);

    float3 sample_dirs[SSMM_MAX_SPP];
    float3 sample_weights[SSMM_MAX_SPP];
    float4 vmfs[SSMM_MAX_SPP];

    mc_state_t curr = mc_state_new(); // zero current state
    mc_state_t tent = mc_state_new(); // tentative new state
    mc_state_t ptent = mc_state_new(); // tentative new state

    const Hit first_hit = params.gbuffer.get_hit(pixel);

    if (first_hit.is_valid() && any(params.gbuffer.get_albedo(pixel).reflection > float3(0))) {
        const RayDesc primary_ray = scene.generate_camera_ray(pixel, resolution);
        const ShadingPoint sp = scene.get_shading_point(primary_ray, first_hit);

        if (sp.has_surface) {
            let material = sp.material;
            const OrthonormalFrame frame = sp.shading_frame;
            let bsdf = material.get_bsdf();
            const float3 wi = -primary_ray.Direction;
            const float3 wi_local = frame.to_local(wi);
            const float3 normal = frame.to_world(float3(0, 0, 1));
            const float3 geonormal = sp.get_face_normal();

            for (int s = 0; s < merian_ssmm_spp; s++) {
                mc_state_shuffle(tent, (WaveGetLaneIndex() + 1) % WaveGetLaneCount());
                sample_dirs[s] = float3(0);
                sample_weights[s] = float3(0);

                read_neighbour_state(params.gbuffer, scene.camera.position, resolution, tent, sp.pos,
                                     normal, int2(pixel), mv, rng);

                vmfs[s] = float4(0);
                if (mc_state_valid(tent))
                    vmfs[s] = mc_state_vmf(tent, sp.pos);

                float pdf = 0;
                float3 directContrib = float3(0);
                float3 direction = float3(0);
                float3 position = float3(0);
                bool bsdf_sample = false;
                do {
                    if (vmfs[s].w == 0 || rng.next_float() < merian_ssmm_bsdf_p) {
                        let wo_local = bsdf.sample(wi_local, rng);
                        if (!wo_local.hasValue)
                            break;
                        direction = frame.to_world(wo_local.value);
                        if (dot(direction, normal) <= 1e-3 || dot(direction, geonormal) <= 1e-3)
                            break;

                        pdf = bsdf.pdf(wi_local, wo_local.value);
                        bsdf_sample = true;
                    } else {
                        direction = vmf_sample_tokuyosh(vmfs[s].xyz, vmfs[s].w, rng.next_float2());
                        pdf = vmf_pdf_tokuyoshi(direction, vmfs[s].xyz, vmfs[s].w);
                    }

                    const float3 throughput = bsdf.eval(wi_local, frame.to_local(direction), rng);

                    RayDesc ray;
                    // Pull back the ray such that it cannot excape through corners (and to prevent self collision)
                    ray.Origin = sp.pos + wi * 1e-3;
                    ray.Direction = direction;
                    ray.TMin = SCENE_RAY_TMIN;
                    ray.TMax = SCENE_RAY_TMAX;
                    const ShadingPoint next_sp =
                        scene.trace_and_get_shading_point(ray, 0xffu, RenderContext(uint16_t(1), 1.0f));

                    position = next_sp.pos;
                    directContrib = throughput * next_sp.emission / pdf;

                    sample_dirs[s] = direction;
                    sample_weights[s] = throughput * next_sp.emission;
                } while (false);

                const float tent_f = yuv_luminance(directContrib);
                if (curr.f == 0 || rng.next_float() < tent_f / curr.f) {
                    if (bsdf_sample)
                        tent = mc_state_new();
                    tent.f = tent_f;
                    mc_state_add(tent, sp.pos, tent_f, direction, position);
                    //tent.add(sd.posW, tent_f, direction, position);
                    curr = tent;
                } else if (!bsdf_sample) {
                    mc_state_add(tent, sp.pos, tent_f, direction, position);
                    // tent.add(sd.posW, tent_f, direction, position);
                }
            }

            for (int s = 0; s < merian_ssmm_spp; s++) {
                if (all(sample_weights[s] == float3(0)))
                    continue;

                const float bsdf_pdf = bsdf.pdf(wi_local, frame.to_local(sample_dirs[s]));
                float sum_pdf = 0;
                for (int t = 0; t < merian_ssmm_spp; t++) {
                    sum_pdf += vmfs[t].w > 0 ? vmf_pdf_tokuyoshi(sample_dirs[s], vmfs[t].xyz, vmfs[t].w)
                                             : bsdf_pdf;
                }

                sum_pdf = merian_ssmm_bsdf_p * merian_ssmm_smis_group_size * bsdf_pdf +
                          (1 - merian_ssmm_bsdf_p) * sum_pdf;

                if (sum_pdf > 0) {
                    const float3 con = sample_weights[s] / sum_pdf;
                    if (!any(isinf(con)) && !any(isnan(con))) {
                        irr += con;
                        const float l = yuv_luminance(con);
                        moments += float2(l, l * l);
                    }
                }
            }

            pc.ssmc_out[ssmm_index(int2(pixel), resolution)] = curr;
        }
    }

    params.irradiance[pixel] =
        float4(irr / max(merian_ssmm_spp, 1), moments.y / max(merian_ssmm_spp, 1));
}
