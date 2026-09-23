import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.colors.colorspaces;
import merian_shaders.colors.colormaps;
import merian_shaders.gbuffer;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"
#include "merian-graph/nodes/path_debug/map_display.slangh"

ParameterBlock<GBuffer> gbuffer;
Sampler2D<float4> in_reference;
Texture2D<float4> in_backdrop;

float3 filtered_tonemapped(const float3 radiance) {
    return params.filtered_exposure > 0.f ? 1.f - exp(-radiance * params.filtered_exposure)
                                          : radiance;
}

bool left_of_split(const uint2 pixel) {
    return float(pixel.x) < params.backdrop_split * float(params.dim_x);
}

float3 sample_reference(const uint2 pixel) {
    return in_reference.SampleLevel((float2(pixel) + 0.5f) / float2(path_debug_dim()), 0.f).rgb;
}

float3 backdrop_color(const uint2 pixel,
                      const float3 render,
                      const float3 filtered_a,
                      const float3 filtered_b) {
    switch (params.backdrop) {
    case PATH_DEBUG_BACKDROP_FILTERED_A:
        return filtered_tonemapped(filtered_a);
    case PATH_DEBUG_BACKDROP_FILTERED_B:
        return filtered_tonemapped(filtered_b);
    case PATH_DEBUG_BACKDROP_SPLIT_FILTERED:
        return filtered_tonemapped(left_of_split(pixel) ? filtered_a : filtered_b);
    case PATH_DEBUG_BACKDROP_REFERENCE:
        return sample_reference(pixel);
    case PATH_DEBUG_BACKDROP_DIFFERENCE:
        return abs(render - sample_reference(pixel)) * params.difference_gain;
    case PATH_DEBUG_BACKDROP_SPLIT_REFERENCE:
        return left_of_split(pixel) ? render : sample_reference(pixel);
    case PATH_DEBUG_BACKDROP_INPUT:
        return in_backdrop[pixel].rgb;
    default:
        return render;
    }
}

float heat_lookup(const float3 pos) {
    if (params.heat_smooth == 0) {
        return path_debug_grid_weight_at(pos);
    }
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

    // normalized by the mean recorded paths per pixel
    const uint base = 4 * (pixel.y * params.dim_x + pixel.x);
    const uint slot_stride = path_debug_filtered_slot_stride();
    float3 filtered_a = float3(0.f);
    float3 filtered_b = float3(0.f);
    if (params.filtered_active != 0) {
        const float recorded = asfloat(filtered[path_debug_filtered_paths_at()]);
        const float paths_per_pixel = recorded / (float(params.dim_x) * float(params.dim_y));
        const float norm = paths_per_pixel > 0.f ? 1.f / paths_per_pixel : 0.f;
        filtered_a = norm * float3(asfloat(filtered[base + 0]), asfloat(filtered[base + 1]),
                                   asfloat(filtered[base + 2]));
        filtered_b = norm * float3(asfloat(filtered[slot_stride + base + 0]),
                                   asfloat(filtered[slot_stride + base + 1]),
                                   asfloat(filtered[slot_stride + base + 2]));
        const float lum_a = yuv_luminance(filtered_a);
        const float lum_b = yuv_luminance(filtered_b);
        out_filtered[pixel] = float4(filtered_a, lum_a * lum_a);
        out_filtered_b[pixel] = float4(filtered_b, lum_b * lum_b);
    }

    const float4 src = in_src[pixel];

    float3 color = backdrop_color(pixel, src.rgb, filtered_a, filtered_b);

    if (params.backdrop == PATH_DEBUG_BACKDROP_SPLIT_REFERENCE ||
        params.backdrop == PATH_DEBUG_BACKDROP_SPLIT_FILTERED) {
        const uint split_x = uint(params.backdrop_split * float(params.dim_x));
        if (pixel.x == split_x || pixel.x + 1 == split_x) {
            color = float3(1.f, 1.f, 0.1f);
        }
    }

    if (params.pixel_stats_view != PATH_DEBUG_PIXEL_STAT_OFF) {
        const PixelMoments m = path_debug_pixel_moments(pixel);
        if (m.count > 1) {
            float value = m.mean;
            if (params.pixel_stats_view == PATH_DEBUG_PIXEL_STAT_REL_ERROR) {
                value = m.rel_error;
            } else if (params.pixel_stats_view == PATH_DEBUG_PIXEL_STAT_VARIANCE) {
                value = m.variance;
            }
            color = lerp(color, colormap_turbo(saturate(value * params.pixel_stats_scale)), 0.75f);
        }
    }

    if (merian_path_debug_gbuffer && params.heat_mode != PATH_DEBUG_HEAT_OFF) {
        const float linear_z = gbuffer.get_linear_z(pixel);
        if (linear_z > 0.f && linear_z < MAX_LINEAR_Z) {
            const float3 pos =
                camera.position_from_linear_z(pixel, path_debug_dim(), linear_z);
            const float w = heat_lookup(pos);
            const float wmax = asfloat(state[PATH_DEBUG_STATE_HEAT_MAX]);
            if (w > 0.f && wmax > 0.f) {
                color = lerp(color,
                             colormap_turbo(path_debug_log_decades(w, wmax, params.heat_exposure)),
                             params.heat_opacity);
            }
        }
    }

    if (merian_path_debug_gbuffer && params.sphere_channel != PATH_DEBUG_MAP_NONE) {
        const float sphere_z = gbuffer.get_linear_z(path_debug_picked_pixel());
        if (sphere_z > 0.f && sphere_z < MAX_LINEAR_Z) {
            const float3 center =
                camera.position_from_linear_z(path_debug_picked_pixel(), path_debug_dim(),
                                              sphere_z);
            const float radius = params.sphere_radius * sphere_z;
            const float3 rd = camera.get_pinhole_direction(pixel, path_debug_dim());
            const float3 oc = camera.position - center;
            const float b = dot(oc, rd);
            const float h2 = b * b - (dot(oc, oc) - radius * radius);
            const float t = -b - sqrt(max(h2, 0.f));
            if (h2 >= 0.f && t >= 0.f) {
                const float3 dir = normalize(camera.position + rd * t - center);
                const float3x3 to_map =
                    dir_map_frame(gbuffer.get_normal(path_debug_picked_pixel()),
                                  params.map_frame == PATH_DEBUG_FRAME_LOCAL);
                float3 sphere_color =
                    path_debug_map_color_dir(params.sphere_channel, mul(to_map, dir));
                sphere_color *= 0.35f + 0.65f * saturate(-dot(dir, rd));
                color = lerp(color, sphere_color, params.overlay_alpha);
            }
        }
    }

    if (path_debug_restricts_region()) {
        const float3 dir = camera.get_pinhole_direction(pixel, path_debug_dim());
        const float3 to_region =
            float3(params.sel_region_x, params.sel_region_y, params.sel_region_z) - camera.position;
        const float along = dot(to_region, dir);
        const float radius = params.sel_region_radius;
        const float half_width = max(
            0.06f * radius, 1.5f * path_debug_pixel_footprint(path_debug_dim() / 2, along));
        if (along > 0.f && abs(length(to_region - along * dir) - radius) < half_width) {
            color = lerp(color, float3(0.35f, 0.85f, 1.f), 0.7f);
        }
    }

    if (path_debug_has_pixel()) {
        const int2 d = abs(int2(pixel) - int2(path_debug_picked_pixel()));
        const bool crosshair = (d.x == 0 && d.y > 2 && d.y < 12) ||
                               (d.y == 0 && d.x > 2 && d.x < 12);
        if (crosshair) {
            color = float3(1.f, 1.f, 0.1f);
        }
    }

    out_out[pixel] = float4(color, src.a);
}
