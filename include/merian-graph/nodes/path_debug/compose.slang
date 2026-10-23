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
Sampler2D<float4> in_backdrop;

static const float3 SPLIT_COLOR = float3(1.f, 1.f, 0.1f);

float3 filtered_tonemapped(const float3 radiance) {
    return params.filtered_exposure > 0.f ? 1.f - exp(-radiance * params.filtered_exposure)
                                          : radiance;
}

float2 out_uv(const uint2 pixel) {
    return (float2(pixel) + 0.5f) / float2(path_debug_out_dim());
}

uint split_x() {
    return uint(params.backdrop_split * float(params.out_x));
}

bool left_of_split(const uint2 pixel) {
    return pixel.x < split_x();
}

float3 sample_reference(const uint2 pixel) {
    return in_reference.SampleLevel(out_uv(pixel), 0.f).rgb;
}

float3 backdrop_color(const uint2 pixel, const float3 render, const float3 filtered) {
    switch (params.backdrop) {
    case PATH_DEBUG_BACKDROP_FILTERED:
        return filtered_tonemapped(filtered);
    case PATH_DEBUG_BACKDROP_REFERENCE:
        return sample_reference(pixel);
    case PATH_DEBUG_BACKDROP_DIFFERENCE:
        return abs(render - sample_reference(pixel)) * params.difference_gain;
    case PATH_DEBUG_BACKDROP_SPLIT_REFERENCE:
        return left_of_split(pixel) ? render : sample_reference(pixel);
    case PATH_DEBUG_BACKDROP_INPUT:
        return in_backdrop.SampleLevel(out_uv(pixel), 0.f).rgb;
    default:
        return render;
    }
}

float heat_lookup(const float3 pos) {
    if (params.heat_smooth == 0) {
        return path_debug_grid_weight_at(pos);
    }
    // the cell and its six face neighbors
    float sum = path_debug_grid_weight_at(pos);
    for (int axis = 0; axis < 3; axis++) {
        for (int side = -1; side <= 1; side += 2) {
            float3 offset = float3(0);
            offset[axis] = float(side) * params.heat_cell_size;
            sum += path_debug_grid_weight_at(pos + offset);
        }
    }
    return sum / 7.f;
}

// normalized by the mean recorded paths per pixel
float3 filtered_radiance(const uint2 render_pixel) {
    if (params.filtered_active == 0) {
        return float3(0.f);
    }
    const float recorded = asfloat(filtered[path_debug_filtered_paths_at()]);
    const float paths_per_pixel = recorded / (float(params.dim_x) * float(params.dim_y));
    const float norm = paths_per_pixel > 0.f ? 1.f / paths_per_pixel : 0.f;
    return norm * path_debug_load_float3(filtered, path_debug_filtered_base(render_pixel));
}

// the scene's distance at a displayed pixel, bilinear between the record pixels
float scene_distance(const uint2 pixel) {
    const float2 at = path_debug_render_coord(pixel) - 0.5f;
    const int2 base = int2(floor(at));
    const float2 f = at - float2(base);
    const int2 last = int2(path_debug_dim()) - 1;
    const float z00 = gbuffer.get_linear_z(uint2(clamp(base, 0, last)));
    const float z10 = gbuffer.get_linear_z(uint2(clamp(base + int2(1, 0), 0, last)));
    const float z01 = gbuffer.get_linear_z(uint2(clamp(base + int2(0, 1), 0, last)));
    const float z11 = gbuffer.get_linear_z(uint2(clamp(base + int2(1, 1), 0, last)));
    return lerp(lerp(z00, z10, f.x), lerp(z01, z11, f.x), f.y);
}

// the displayed pixel at the center of the cursor's record pixel
uint2 cursor_display_pixel() {
    if (params.cursor_x == PATH_DEBUG_NO_PIXEL) {
        return uint2(PATH_DEBUG_NO_PIXEL);
    }
    return uint2((float2(params.cursor_x, params.cursor_y) + 0.5f) * float2(path_debug_out_dim()) /
                 float2(path_debug_dim()));
}

float3 compose(const uint2 pixel, const float3 src) {
    const uint2 render_pixel = path_debug_render_pixel(pixel);
    float3 color = backdrop_color(pixel, src, filtered_radiance(render_pixel));

    if (params.backdrop == PATH_DEBUG_BACKDROP_SPLIT_REFERENCE) {
        if (pixel.x == split_x() || pixel.x + 1 == split_x()) {
            color = SPLIT_COLOR;
        }
    }

    if (params.pixel_stats_view != PATH_DEBUG_PIXEL_STAT_OFF) {
        const PathDebugPixelMoments m = path_debug_pixel_moments(render_pixel);
        if (m.count > 1) {
            float value = m.mean;
            if (params.pixel_stats_view == PATH_DEBUG_PIXEL_STAT_REL_ERROR) {
                value = m.rel_error;
            } else if (params.pixel_stats_view == PATH_DEBUG_PIXEL_STAT_VARIANCE) {
                value = m.variance;
            }
            color = lerp(color, colormap_turbo(saturate(value * params.pixel_stats_scale)),
                         params.false_color_opacity);
        }
    }

    if (merian_path_debug_gbuffer && params.heat_mode != PATH_DEBUG_HEAT_OFF) {
        const float linear_z = gbuffer.get_linear_z(render_pixel);
        if (linear_z > 0.f && linear_z < MAX_LINEAR_Z) {
            const float3 pos =
                camera.position_from_linear_z(render_pixel, path_debug_dim(), linear_z);
            const float w = heat_lookup(pos);
            const float wmax = asfloat(state[PATH_DEBUG_STATE_HEAT_MAX]);
            if (w > 0.f && wmax > 0.f) {
                color = lerp(color,
                             colormap_turbo(path_debug_log_decades(w, wmax, params.heat_exposure)),
                             params.false_color_opacity);
            }
        }
    }

    if (merian_path_debug_gbuffer && params.sphere_channel != PATH_DEBUG_MAP_NONE &&
        params.sel_point_valid != 0) {
        const float3 center = path_debug_selected_point();
        const float radius = params.sphere_radius * distance(camera.position, center);
        const float3 rd =
            camera.get_pinhole_direction_unjittered(float2(pixel), path_debug_out_dim());
        const float3 oc = camera.position - center;
        const float b = dot(oc, rd);
        const float h2 = b * b - (dot(oc, oc) - radius * radius);
        const float t = -b - sqrt(max(h2, 0.f));
        // the scene in front of it hides it
        if (h2 >= 0.f && t >= 0.f && t < scene_distance(pixel)) {
            const float3 dir = normalize(camera.position + rd * t - center);
            const float3 selected_normal = gbuffer.get_normal(path_debug_selected_pixel());
            const float3x3 to_map = dir_map_frame(path_debug_map_normal(selected_normal),
                                                  params.map_frame == PATH_DEBUG_FRAME_LOCAL);
            const float3 map_dir = mul(to_map, dir);
            const float3 uvw = dir_map_uv(params.map_transform, map_dir);
            if (uvw.z != 0.f && all(pixel == cursor_display_pixel())) {
                state[PATH_DEBUG_STATE_SPHERE_PROBE] =
                    path_debug_map_bin(path_debug_map_cell(uvw.xy)) + 1;
            }
            const float3 sphere_color = path_debug_map_color_dir(params.sphere_channel, map_dir) *
                                        (0.35f + 0.65f * saturate(-dot(dir, rd)));
            color = lerp(color, sphere_color, params.overlay_alpha);
        }
    }
    return color;
}

// One thread per pixel of the larger of the displayed image and the record pixels.
[numthreads(PATH_DEBUG_WG_2D, PATH_DEBUG_WG_2D, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint2 pixel = gid.xy;

    if (all(pixel < path_debug_dim())) {
        if (params.filtered_out != 0) {
            const float3 filtered = filtered_radiance(pixel);
            const float lum = yuv_luminance(filtered);
            out_filtered[pixel] = float4(filtered, lum * lum);
        }
        if (merian_path_debug_gbuffer && params.pick_x != PATH_DEBUG_NO_PIXEL &&
            all(pixel == uint2(params.pick_x, params.pick_y))) {
            const float linear_z = gbuffer.get_linear_z(pixel);
            const float3 surface = camera.position_from_linear_z(
                pixel, path_debug_dim(),
                linear_z < MAX_LINEAR_Z ? linear_z : PATH_DEBUG_ENV_DISTANCE);
            const float3 n = gbuffer.get_normal(pixel);
            const float3 normal = all(isfinite(n)) ? n : float3(0.f);
            path_debug_store_float3(state, PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_SURFACE,
                                    surface);
            path_debug_store_float3(state, PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_NORMAL, normal);
            state[PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_SURFACE_SEQ] = params.pick_seq;
        }
    }

    if (all(pixel < path_debug_out_dim())) {
        const float4 src = in_src.SampleLevel(out_uv(pixel), 0.f);
        out_out[pixel] = float4(compose(pixel, src.rgb), src.a);
    }
}
