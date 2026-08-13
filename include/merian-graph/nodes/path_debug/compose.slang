import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.textures;
import merian_shaders.colors.colorspaces;
import merian_shaders.colors.colormaps;
import merian_shaders.gbuffer;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

ParameterBlock<GBuffer> gbuffer;
// A/B reference; a 1x1 black texture while no image is loaded
Sampler2D<float4> in_reference;

// The image the debug overlays sit on: the input, the paths the filter kept, or a reference
// image loaded from disk (whole, side by side, or as an amplified difference).
float3 filtered_tonemapped(const float3 filtered_color) {
    // raw radiance: the graph's exposure and tonemap sit before this node
    return params.ab_scale > 0.f ? 1.f - exp(-filtered_color * params.ab_scale) : filtered_color;
}

float3 base_color(const uint2 pixel,
                  const float3 src,
                  const float3 filtered_a,
                  const float3 filtered_b) {
    const float3 reference =
        in_reference.SampleLevel((float2(pixel) + 0.5f) / float2(path_debug_dim()), 0.f).rgb;

    switch (params.render_mode) {
    case PATH_DEBUG_RENDER_FILTERED:
        return filtered_tonemapped(filtered_a);
    case PATH_DEBUG_RENDER_FILTERED_B:
        return filtered_tonemapped(filtered_b);
    case PATH_DEBUG_RENDER_SPLIT_FILTERS:
        return filtered_tonemapped(float(pixel.x) < params.ab_split * float(params.dim_x)
                                       ? filtered_a
                                       : filtered_b);
    case PATH_DEBUG_RENDER_REFERENCE:
        return reference;
    case PATH_DEBUG_RENDER_DIFF:
        return abs(src - reference) * params.ab_scale;
    case PATH_DEBUG_RENDER_SPLIT:
        return float(pixel.x) < params.ab_split * float(params.dim_x) ? src : reference;
    default:
        return src;
    }
}

float heat_lookup(const float3 pos) {
    if (params.heat_smooth == 0) {
        return path_debug_grid_weight_at(pos);
    }
    // center + face neighbors
    float sum = path_debug_grid_weight_at(pos);
    [unroll]
    for (int axis = 0; axis < 3; axis++) {
        [unroll]
        for (int side = -1; side <= 1; side += 2) {
            float3 offset = float3(0);
            offset[axis] = float(side) * params.heat_cell_size;
            sum += path_debug_grid_weight_at(pos + offset);
        }
    }
    return sum / 7.f;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint2 pixel = gid.xy;
    if (any(pixel >= path_debug_dim())) {
        return;
    }

    // Radiance of the paths the expression kept, published for a downstream accumulate chain.
    // Normalized by the frame's average paths per pixel (= spp * keep probability), so the
    // estimate stays unbiased under subsampling; the realized per-pixel count would ignore the
    // frames that recorded nothing.
    const uint base = 4 * (pixel.y * params.dim_x + pixel.x);
    const uint slot_stride = 4 * params.dim_x * params.dim_y;
    const uint recorded = min(in_records[PATH_RECORD_HDR_PATH_COUNT],
                              path_record_path_capacity(in_records));
    const float paths_per_pixel =
        float(recorded) / (float(params.dim_x) * float(params.dim_y));
    const float norm = paths_per_pixel > 0.f ? 1.f / paths_per_pixel : 0.f;
    const float3 filtered_a = norm * float3(asfloat(filtered[base + 0]),
                                            asfloat(filtered[base + 1]),
                                            asfloat(filtered[base + 2]));
    const float3 filtered_b = norm * float3(asfloat(filtered[slot_stride + base + 0]),
                                            asfloat(filtered[slot_stride + base + 1]),
                                            asfloat(filtered[slot_stride + base + 2]));
    // alpha carries the second moment, matching the renderers' irradiance convention
    const float lum_a = yuv_luminance(filtered_a);
    const float lum_b = yuv_luminance(filtered_b);
    out_filtered[pixel] = float4(filtered_a, lum_a * lum_a);
    out_filtered_b[pixel] = float4(filtered_b, lum_b * lum_b);

    const float4 src = in_src[pixel];

    if (params.ab_metrics != 0) {
        const float3 reference =
            in_reference.SampleLevel((float2(pixel) + 0.5f) / float2(path_debug_dim()), 0.f).rgb;
        const float3 diff = src.rgb - reference;
        const float relsq =
            dot(diff * diff / (reference * reference + 1e-4f), float3(1.f / 3.f));
        const float signed_diff = yuv_luminance(src.rgb) - yuv_luminance(reference);
        const float wave_relsq = WaveActiveSum(relsq);
        const float wave_signed = WaveActiveSum(signed_diff);
        const uint wave_n = WaveActiveCountBits(true);
        if (WaveIsFirstLane()) {
            path_debug_atomic_add_float(stats, PATH_DEBUG_STATS_AB_RELSQ, wave_relsq);
            path_debug_atomic_add_float(stats, PATH_DEBUG_STATS_AB_SIGNED, wave_signed);
            InterlockedAdd(stats[PATH_DEBUG_STATS_AB_N], wave_n);
        }
    }
    float3 color = base_color(pixel, src.rgb, filtered_a, filtered_b);

    if (params.render_mode == PATH_DEBUG_RENDER_SPLIT ||
        params.render_mode == PATH_DEBUG_RENDER_SPLIT_FILTERS) {
        const uint split_x = uint(params.ab_split * float(params.dim_x));
        if (pixel.x == split_x || pixel.x + 1 == split_x) {
            color = float3(1.f, 1.f, 0.1f);
        }
    }

    if (params.error_view != 0) {
        const uint m = path_debug_moments_at(pixel);
        const float sum = asfloat(moments[m + 0]);
        const float sum2 = asfloat(moments[m + 1]);
        const uint n = moments[m + 2];
        if (n > 1) {
            const float mean = sum / float(n);
            const float variance = max(sum2 / float(n) - mean * mean, 0.f);
            float value = 0.f;
            if (params.error_view == 1) {
                value = mean > 0.f ? sqrt(variance / float(n)) / mean : 0.f;
            } else if (params.error_view == 2) {
                value = variance;
            } else {
                value = mean;
            }
            color = lerp(color, colormap_turbo(saturate(value * params.error_scale)), 0.75f);
        }
    }

    if (merian_path_debug_gbuffer && params.heat_mode != PATH_DEBUG_HEAT_OFF) {
        const float linear_z = gbuffer.get_linear_z(pixel);
        if (linear_z > 0.f && linear_z < 1e6f) {
            const float3 dir = camera.get_pinhole_direction(pixel, path_debug_dim());
            const float3 pos = camera.position + dir * linear_z;
            const float w = heat_lookup(pos);
            const float wmax = asfloat(state[PATH_DEBUG_STATE_HEAT_MAX]);
            if (w > 0.f && wmax > 0.f) {
                // scale-free: three decades below the max, shifted by exposure
                const float t = log(1.f + params.heat_exposure * 999.f * w / wmax) / log(1000.f);
                color = lerp(color, colormap_turbo(saturate(t)), params.heat_opacity);
            }
        }
    }

    const uint packed = overlay[pixel.y * params.dim_x + pixel.x];
    if (packed != 0) {
        color = lerp(color, path_debug_unpack_overlay(packed), params.overlay_alpha);
    }

    if (params.mode == PATH_DEBUG_MODE_SELECTED_PIXEL) {
        const int2 d = abs(int2(pixel) - int2(path_debug_selected_pixel()));
        const bool crosshair = (d.x == 0 && d.y > 2 && d.y < 12) ||
                               (d.y == 0 && d.x > 2 && d.x < 12);
        if (crosshair) {
            color = float3(1.f, 1.f, 0.1f);
        }
    }

    out_out[pixel] = float4(color, src.a);
}
