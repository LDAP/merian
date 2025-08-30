
#include "config.h"
#include "svgf_graph_layout.slangh"

import merian_shaders.utils.textures;
import merian_shaders.utils.shared_memory;
import merian_shaders.colors.colorspaces;

using merian;

/*
 * Reads the accumulated irradiance and the moments, estimates the variance,
 * and prepares the filter input `img_filter_in`.
 */

[vk::constant_id(2)]
const int SVGF_ITERATIONS = 0;

[vk::binding(0, 1)]
Sampler2D<float4> img_filter_unused; // unused needed to match the filter layout
[vk::binding(1, 1)]
RWTexture2D<float4> img_filter_in;
[vk::binding(2, 1)]
Sampler2D<uint4> img_gbuf_unused; // unused needed to match the filter layout
[vk::binding(3, 1)]
RWTexture2D<uint4> img_gbuf_in;

struct PushConstant {
    float normal_reject_cos;
    float depth_accept; /* -10 / depth_accept precomputed */
    float spatial_falloff;
    float spatial_bias;
};

[vk::push_constant]
ConstantBuffer<PushConstant> params;

// update VE_SHARED_MEMORY_PER_PIXEL if you change anything here!
#define SHARED_SIZE_X (workgroup_size_x + SVGF_VE_HALO_RADIUS * 2)
#define SHARED_SIZE_Y (workgroup_size_y + SVGF_VE_HALO_RADIUS * 2)

row_major groupshared float4 shared_irr_moments[SHARED_SIZE_Y][SHARED_SIZE_X];
row_major groupshared float4 shared_normal_z[SHARED_SIZE_Y][SHARED_SIZE_X];

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(const int2 ipos: SV_DispatchThreadID, int2 gl_LocalInvocationID: SV_GroupThreadID) {
    if (SVGF_ITERATIONS == 0) {
        if (is_on_texture(ipos, img_filter_in))
            img_filter_in.Store(ipos, float4(img_irr.Load(uint3(ipos, 0)).rgb, 0));
        return;
    }

    const int2 shared_pos = gl_LocalInvocationID + SVGF_VE_HALO_RADIUS;

    // Load all necessary information into shared memory

    half2 center_grad_z;
    float4 irr_moments;

#define center_irr_moments shared_irr_moments[shared_pos.y][shared_pos.x]
#define center_normal shared_normal_z[shared_pos.y][shared_pos.x].rgb
#define center_linear_z shared_normal_z[shared_pos.y][shared_pos.x].a
#define center_normal_z shared_normal_z[shared_pos.y][shared_pos.x]

    // fill in padding and halo with valid information
    const int2 mirror_repeat_ipos = texturewrap_mirror_repeat(ipos, texture_dimensions(img_irr, 0));

    {
        {
            // center
            {
                const GBuffer gbuf = GBuffer::load(gbuffer, mirror_repeat_ipos);
                irr_moments = img_irr.Load(uint3(mirror_repeat_ipos, 0));
                irr_moments.rgb = rgb_to_yuv(irr_moments.rgb);

                center_irr_moments = irr_moments;
                center_normal_z = float4(gbuf.get_normal(), gbuf.get_linear_z());
                center_grad_z = gbuf.get_grad_z();
            }

            // halo
            int2 shared_index;
            int2 global_index;
            if (load_halo_index(SVGF_VE_HALO_RADIUS, WorkgroupSize().xy, shared_index,
                                global_index)) {
                const int2 halo_mirror_repeat_ipos =
                    texturewrap_mirror_repeat(global_index, texture_dimensions(img_irr, 0));
                const GBuffer gbuf = GBuffer::load(gbuffer, halo_mirror_repeat_ipos);

                shared_irr_moments[shared_index.y][shared_index.x] =
                    img_irr.Load(uint3(halo_mirror_repeat_ipos, 0));
                shared_normal_z[shared_index.y][shared_index.x] =
                    float4(gbuf.get_normal(), gbuf.get_linear_z());
            }

            GroupMemoryBarrierWithGroupSync();
        }

        if (not_on_texture(ipos, img_filter_in))
            return;

#ifdef KALEIDOSCOPE
        img_gbuf_in.Store(ipos, uint4(encode_normal(center_normal), asuint(center_linear_z),
                                      packHalf2x16(center_grad_z), 0));
#endif
    }

    float sum_w = 1.0;

    const float kernel[] = { 1., 2. / 3., 1. / 6. };

    // spatial filtering when variance is 0 leads to unnecessary filtering of surfaces which
    // variance is really 0 however it prevents black trails.
    [unroll]
    for (int yy = -SVGF_VE_HALO_RADIUS; yy <= SVGF_VE_HALO_RADIUS; yy++) {
        [unroll]
        for (int xx = -SVGF_VE_HALO_RADIUS; xx <= SVGF_VE_HALO_RADIUS; xx++) {
            if (xx != 0 || yy != 0) { // added already
                const int2 p = int2(shared_pos.xy) + int2(xx, yy);

                // smoothing kernel
                float w = kernel[abs(xx)] * kernel[abs(yy)];
                // Weight neighbors with similar normals
                w *= smoothstep(params.normal_reject_cos, 1.0,
                                dot(center_normal, shared_normal_z[p.y][p.x].rgb));
                // Weight neighbors with similar depth
                // w *= 1.0 - smoothstep(0.0, params.depth_reject_percent * max(gbuf.linear_z,
                // neigh_gbuf.linear_z), abs(gbuf.linear_z - neigh_gbuf.linear_z));
                w *= exp(params.depth_accept *
                         abs(center_linear_z + dot(float2(center_grad_z), float2(xx, yy)) -
                             shared_normal_z[p.y][p.x].a));
                // Weight lower with increasing history
                // w *= 1.0 - irr_moments.a / params.spatial_threshold;

                sum_w += w;
                irr_moments += w * shared_irr_moments[p.y][p.x];
            }
        }
    }

    irr_moments /= sum_w;

    const float history = img_history.Load(uint3(mirror_repeat_ipos, 0)).r;
    const float alpha = saturate(params.spatial_bias / pow(history, params.spatial_falloff));

    irr_moments = lerp(center_irr_moments, irr_moments, alpha);

    const float variance = max(0.0, irr_moments.a - irr_moments.r * irr_moments.r);
    img_filter_in.Store(ipos, float4(irr_moments.rgb, variance));
}

