#include "config.h"
#include "svgf_graph_layout.slangh"

[vk::binding(0, 1)]
Sampler2D<float4> img_filter_in;
[vk::binding(1, 1)]
RWTexture2D<float4> img_filter_out;
[vk::binding(2, 1)]
Sampler2D<uint4> img_gbuf_in;
[vk::binding(3, 1)]
RWTexture2D<uint4> img_gbuf_out;

[vk::constant_id(2)]
const int GAP = 1;
[vk::constant_id(3)]
const int ITERATION = 0;
[vk::constant_id(4)]
const int LAST_ITERATION = 0;

// the reference implementation does that but it looks better without
// #define FILTER_VARIANCE

// defined by compiler:
// #define KALEIDOSCOPE
// #define KALEIDOSCOPE_USE_SHMEM

// commet to disable; two variants to reconstruct (0, 1) Reiner recommends using 1
// #define KALEIDOSCOPE_USE_MIRRORING 1

struct PushConstant {
    float param_z; /* -10 / param_z precomputed */
    float param_n;
    float param_l;
    float z_bias_normals;
    float z_bias_depth;
}
[vk::push_constant]
ConstantBuffer<PushConstant> params;

import merian_shaders.utils.textures;
import merian_shaders.utils.shared_memory;
import merian_shaders.colors.colorspaces;

using merian;

// --------------------------------------------------------------

static int2 ipos;

#ifdef KALEIDOSCOPE_USE_SHMEM
static int2 ipos_shared;
#endif // KALEIDOSCOPE_USE_SHMEM

static float4 filter_irr_var;
static float filter_w;

static float2 c_grad_z;
static float c_z;
static float3 c_n;

static float c_l;
static float sigma_l;

static float z_bias_depth_mix;
static float z_bias_normals_mix;

// --------------------------------------------------------------

#ifdef FILTER_VARIANCE

float get_sigma_l(float center, int2 ipos) {
    const float gaussian_kernel[3][3] = { { 1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0 },
                                          { 1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0 },
                                          { 1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0 } };

    float sum = center * gaussian_kernel[1][1];
    const int r = 1;
    for (int yy = -r; yy <= r; yy++) {
        for (int xx = -r; xx <= r; xx++) {
            if (xx != 0 || yy != 0) {
                const int2 p = ipos + int2(xx, yy);
                const float v = img_filter_in.Load(int3(p, 0)).a;
                const float w = gaussian_kernel[xx + 1][yy + 1];
                sum += v * w;
            }
        }
    }

    return sqrt(max(sum, 0.0));
}

#endif // FILTER_VARIANCE

#ifdef KALEIDOSCOPE
#ifdef KALEIDOSCOPE_USE_SHMEM

#define WORKGROUP_SIZE_X workgroup_size_x
#define WORKGROUP_SIZE_Y workgroup_size_y

#define SHARED_SIZE_X (WORKGROUP_SIZE_X + SVGF_FILTER_HALO_RADIUS * 2)
#define SHARED_SIZE_Y (WORKGROUP_SIZE_Y + SVGF_FILTER_HALO_RADIUS * 2)

// remember to update config.h if anything is changed here!
// FILTER_SHARED_MEMORY_PER_PIXEL = 2 * 4 * 4
// total shared size: SHARED_SIZE_X * SHARED_SIZE_Y * FILTER_SHARED_MEMORY_PER_PIXEL

// (irr, var)
row_major groupshared float4[SHARED_SIZE_Y][SHARED_SIZE_X] SHARED_TILE_u32vec4_A;
// (normal, linear_z)
row_major groupshared float4[SHARED_SIZE_Y][SHARED_SIZE_X] SHARED_TILE_u32vec4_B;

void shmem_read(const int2 pos, out float3 n, out float z, out float4 irrfilter) {
    irrfilter = SHARED_TILE_u32vec4_A[pos.y][pos.x];
    n = SHARED_TILE_u32vec4_B[pos.y][pos.x].rgb;
    z = SHARED_TILE_u32vec4_B[pos.y][pos.x].a;
}

void shmem_read(const int2 pos, out float4 gbuf, out float4 irrfilter) {
    irrfilter = SHARED_TILE_u32vec4_A[pos.y][pos.x];
    gbuf = SHARED_TILE_u32vec4_B[pos.y][pos.x];
}

void shmem_write(const int2 pos, const float4 irrfilter, const float3 n, const float z) {
    SHARED_TILE_u32vec4_A[pos.y][pos.x] = irrfilter;
    SHARED_TILE_u32vec4_B[pos.y][pos.x] = float4(n, z);
}

#endif // KALEIDOSCOPE_USE_SHMEM
#endif // KALEIDOSCOPE

// --------------------------------------------------------------

#ifdef KALEIDOSCOPE
#define tap(offset, weight) _tap(offset, weight)
#else
#define tap(offset, weight) _tap(offset* GAP, weight)
#endif

void _tap(const int2 offset, const float kernel_weight) {

#ifdef KALEIDOSCOPE
#ifdef KALEIDOSCOPE_USE_SHMEM
    float4 p_irr_var;
    float3 p_n;
    float p_z;
    shmem_read(ipos_shared + offset, p_n, p_z, p_irr_var);
#else
    const float4 p_irr_var = img_filter_in.Load(int3(ipos + offset, 0));
    const float3 p_n = decode_normal(img_gbuf_in.Load(int3(ipos + offset, 0)).x);
    const float p_z = asfloat(img_gbuf_in.Load(int3(ipos + offset, 0)).y);
#endif

#define grad_offset (offset * GAP)
#else
    const float4 p_irr_var = img_filter_in.Load(int3(ipos + offset, 0));
    const float3 p_n = GBuffer.load_and_decompress_normal(gbuffer, ipos + offset);
    const float p_z = GBuffer.load_and_decompress_linear_z(gbuffer, ipos + offset);
#define grad_offset offset
#endif

    const float w_l = abs(p_irr_var.r - c_l) * sigma_l; // we use yuv in irr, so r is luminance
    const float w_z = exp(params.param_z * abs(c_z + dot(c_grad_z, grad_offset) - p_z));
    const float w_n = smoothstep(params.param_n, 1.0, dot(p_n, c_n));
    // was: pow(max(0, dot(p_n, c_n)), params.param_n);

    const float w = exp(-w_l * w_l) * lerp(w_z, 1.0, z_bias_depth_mix) *
                    lerp(w_n, 1.0, z_bias_normals_mix) * kernel_weight;

    filter_irr_var += p_irr_var * float4(w, w, w, w * w);
    filter_w += w;

#undef grad_offset
}

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(const int2 dispatch_thread_id: SV_DispatchThreadID,
          const int2 group_thread_id: SV_GroupThreadID,
          const uint group_index: SV_GroupIndex,
          const uint2 group_id: SV_GroupID) {

#ifdef KALEIDOSCOPE
#ifdef KALEIDOSCOPE_USE_SHMEM

    ipos = int2(dispatch_thread_id);
    ipos_shared = int2(group_thread_id) + SVGF_FILTER_HALO_RADIUS;

    filter_irr_var = img_filter_in.Load(int3(ipos, 0));
    const uint3 c_gbuf = img_gbuf_in.Load(int3(ipos, 0)).rgb;
    c_n = decode_normal(c_gbuf.x);
    c_z = asfloat(c_gbuf.y);
    c_grad_z = unpackHalf2x16ToFloat(c_gbuf.z);

    {
        // center
        shmem_write(ipos_shared, filter_irr_var, c_n, c_z);
        // halo
        int2 shared_index;
        int2 global_index;
        if (load_halo_index(SVGF_FILTER_HALO_RADIUS, WorkgroupSize().xy, group_id, group_index,
                            shared_index, global_index)) {
            const int2 global_read =
                texturewrap_mirror_repeat(global_index, texture_dimensions(img_filter_in, 0));
            const uint3 p_gbuf = img_gbuf_in.Load(int3(global_read, 0)).rgb;
            shmem_write(shared_index, img_filter_in.Load(int3(global_read, 0)),
                        decode_normal(p_gbuf.x), asfloat(p_gbuf.y));
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (not_on_texture(ipos, img_filter_out))
        return;
#else // NO KALEIDOSCOPE_SHMEM:
    ipos = int2(dispatch_thread_id);
    if (not_on_texture(ipos, img_filter_out))
        return;

    filter_irr_var = img_filter_in.Load(int3(ipos, 0));
    const uint3 c_gbuf = img_gbuf_in.Load(uint3(ipos, 0)).rgb;
    c_n = decode_normal(c_gbuf.x);
    c_z = asfloat(c_gbuf.y);
    c_grad_z = unpackHalf2x16ToFloat(c_gbuf.z);
#endif
#else // NO KALEIDOSCOPE:
    ipos = int2(dispatch_thread_id);
    if (not_on_texture(ipos, img_filter_out))
        return;

    filter_irr_var = img_filter_in.Load(int3(ipos, 0));
    const CompressedGBuffer c_gbuf = GBuffer.load_compressed(gbuffer, ipos);
    c_n = c_gbuf.decompress_normal();
    c_z = c_gbuf.decompress_linear_z();
    c_grad_z = c_gbuf.decompress_grad_z();
#endif

    c_l = filter_irr_var.r;

    if (params.z_bias_normals > 0)
        z_bias_normals_mix = smoothstep(0, params.z_bias_normals, c_z);
    else
        z_bias_normals_mix = 0;

    if (params.z_bias_depth > 0)
        z_bias_depth_mix = smoothstep(0, params.z_bias_depth, c_z);
    else
        z_bias_depth_mix = 0;

#ifdef FILTER_VARIANCE
    sigma_l = 1. / (get_sigma_l(filter_irr_var.a, ipos) * params.param_l + 1e-10);
#else
    sigma_l = 1. / (sqrt(filter_irr_var.a) * params.param_l + 1e-10);
#endif

    if (c_z > 0) {
        filter_w = 1.0;

#if FILTER_TYPE == 0
        // atrous
        const float kernel[] = { 1., 2. / 3., 1. / 6. };
        [unroll]
        for (int j = -2; j <= 2; j++) {
            [unroll]
            for (int i = -2; i <= 2; i++) {
                if (i != 0 || j != 0) {
                    const float weight = kernel[abs(i)] * kernel[abs(j)];
                    tap(int2(i, j), weight);
                }
            }
        }
#elif FILTER_TYPE == 1
        // box3
        const int r = 1;
        [unroll]
        for (int yy = -r; yy <= r; yy++) {
            [unroll]
            for (int xx = -r; xx <= r; xx++) {
                if (xx != 0 || yy != 0) {
                    tap(int2(xx, yy), 1.0);
                }
            }
        }
#elif FILTER_TYPE == 2
        // subsampled
        if ((ITERATION & 1) == 0) {
            tap(int2(-2, 0), 1.0);
            tap(int2(2, 0), 1.0);
        } else {
            tap(int2(0, -2), 1.0);
            tap(int2(0, 2), 1.0);
        }
        tap(int2(-1, 1), 1.0);
        tap(int2(1, 1), 1.0);
        tap(int2(-1, -1), 1.0);
        tap(int2(1, -1), 1.0);
#endif

        filter_irr_var.rgb /= filter_w;
        filter_irr_var.a /= filter_w * filter_w;
    }

    if (ITERATION == LAST_ITERATION) {
        filter_irr_var.rgb = yuv_to_rgb(filter_irr_var.rgb);
    }

// --------------------------------------------------------------
// Reiners Kaleidoscope-SVGF Swizzle
#ifdef KALEIDOSCOPE
    if (ITERATION != LAST_ITERATION) {
        int2 hid = ipos / 2;
        int2 odd = ipos & 1;
        int2 viewport_size_half = texture_dimensions(img_filter_out) / 2;

#ifdef KALEIDOSCOPE_USE_MIRRORING
        if (ITERATION == 0) {
            if (odd.x > 0) {
                hid.x = viewport_size_half.x - hid.x - 1;
            }
            if (odd.y > 0) {
                hid.y = viewport_size_half.y - hid.y - 1;
            }
        }
#endif

        ipos = hid + odd * (viewport_size_half);
        img_gbuf_out.Store(ipos, uint4(c_gbuf, 0));
    } else {
        int2 viewport_size = texture_dimensions(img_filter_out);
        int32_t wcnt = 1 << ITERATION;
        int2 sizew = viewport_size / wcnt;
        int2 o = ipos / sizew;
        int2 wid = ipos - sizew * o;

#if defined(KALEIDOSCOPE_USE_MIRRORING) && KALEIDOSCOPE_USE_MIRRORING == 0
        int2 odd = o & 1;
        if (odd.x > 0) {
            wid.x = sizew.x - wid.x - 1;
            o.x = wcnt - o.x;
        }
        if (odd.y > 0) {
            wid.y = sizew.y - wid.y - 1;
            o.y = wcnt - o.y;
        }
#endif

        ipos = o + wid * wcnt;

#if defined(KALEIDOSCOPE_USE_MIRRORING) && KALEIDOSCOPE_USE_MIRRORING == 1
        if (ITERATION != 0 /* skip if we're doing exactly one iteration */) {
            int2 hid = ipos / 2;
            int2 odd = ipos & 1;
            int2 viewport_size_half = texture_dimensions(img_filter_out) / 2;
            if (odd.x > 0) {
                hid.x = viewport_size_half.x - hid.x - 1;
            }
            if (odd.y > 0) {
                hid.y = viewport_size_half.y - hid.y - 1;
            }
            ipos = hid * 2 + odd;
        }
#endif
    }
#endif

    img_filter_out.Store(ipos, filter_irr_var);
}

