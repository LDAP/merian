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

float3 recorded_tonemapped(const float3 radiance) {
    return 1.f - exp(-radiance * params.image_exposure);
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

float3 backdrop_color(const uint2 pixel, const float3 render, const uint2 render_pixel) {
    switch (params.backdrop) {
    case PATH_DEBUG_BACKDROP_FRAME:
    case PATH_DEBUG_BACKDROP_ACCUMULATED:
    case PATH_DEBUG_BACKDROP_FRAME_FILTERED:
    case PATH_DEBUG_BACKDROP_ACCUMULATED_FILTERED:
        return recorded_tonemapped(
            recorded_radiance(render_pixel, path_debug_backdrop_filtered(params.backdrop)));
    case PATH_DEBUG_BACKDROP_REFERENCE:
        return params.image_exposure * sample_reference(pixel);
    case PATH_DEBUG_BACKDROP_DIFFERENCE:
        return params.image_exposure * params.difference_gain *
               abs(render - sample_reference(pixel));
    case PATH_DEBUG_BACKDROP_SPLIT_REFERENCE:
        return params.image_exposure * (left_of_split(pixel) ? render : sample_reference(pixel));
    case PATH_DEBUG_BACKDROP_INPUT:
        return params.image_exposure * in_backdrop.SampleLevel(out_uv(pixel), 0.f).rgb;
    default:
        return params.image_exposure * render;
    }
}

float heat_lookup(const float3 pos) {
    float2 cell = path_debug_grid_at(pos);
    if (params.heat_smooth == 0) {
        return path_debug_heat_value(cell);
    }
    for (int axis = 0; axis < 3; axis++) {
        for (int side = -1; side <= 1; side += 2) {
            float3 offset = float3(0);
            offset[axis] = float(side) * params.heat_cell_size;
            cell += path_debug_grid_at(pos + offset);
        }
    }
    return path_debug_heat_value(cell / 7.f);
}

float3 recorded_radiance(const uint2 render_pixel, const bool with_filter) {
    if (params.filtered_active == 0) {
        return float3(0.f);
    }
    const uint2 source = path_debug_source_pixel(render_pixel);
    if (source.x == PATH_DEBUG_NO_PIXEL) {
        return float3(0.f);
    }
    const uint base = path_debug_filtered_base(source, with_filter && path_debug_narrows());
    const uint count = filtered[base + PATH_DEBUG_FILTERED_COUNT];
    return count > 0 ? path_debug_load_float3(filtered, base) / float(count) : float3(0.f);
}

// the scene's distance at a displayed pixel, bilinear between the record pixels
float scene_distance(const uint2 pixel) {
    const float2 at = path_debug_render_coord(pixel) - 0.5f;
    const int2 base = int2(floor(at));
    const float2 f = at - float2(base);
    const int2 last = int2(path_debug_dim()) - 1;
    const float z00 = gbuffer.get_first_hit_linear_z(uint2(clamp(base, 0, last)));
    const float z10 = gbuffer.get_first_hit_linear_z(uint2(clamp(base + int2(1, 0), 0, last)));
    const float z01 = gbuffer.get_first_hit_linear_z(uint2(clamp(base + int2(0, 1), 0, last)));
    const float z11 = gbuffer.get_first_hit_linear_z(uint2(clamp(base + int2(1, 1), 0, last)));
    return lerp(lerp(z00, z10, f.x), lerp(z01, z11, f.x), f.y);
}

uint2 cursor_display_pixel() {
    if (params.cursor_x == PATH_DEBUG_NO_PIXEL) {
        return uint2(PATH_DEBUG_NO_PIXEL);
    }
    return uint2((float2(params.cursor_x, params.cursor_y) + 0.5f) * float2(path_debug_out_dim()) /
                 float2(path_debug_dim()));
}

struct SphereHit {
    bool valid;
    float t;
    // from the center to the hit, and the surface normal there
    float3 dir;
    float3 normal;
}

static const uint SPHERE_STEPS = 96;
static const uint SPHERE_REFINES = 8;
// the radius of a direction without value, over the full one
static const float SPHERE_MIN_EXTENT = 0.02f;

float sphere_extent(const float3 dir, const float3x3 to_map) {
    if (params.sphere_scale == 0) {
        return 1.f;
    }
    return max(path_debug_map_extent_dir(params.sphere_channel, mul(to_map, dir)),
               SPHERE_MIN_EXTENT);
}

// The sphere at the selected point; scaled, its radius per direction is the value on the panel's
// color scale, marched inside the full sphere and refined by bisection.
SphereHit trace_sphere(const float3 rd, out float3x3 to_map) {
    SphereHit hit = { false, 0.f, float3(0), float3(0) };
    to_map = float3x3(0.f);
    const float3 center = path_debug_selected_point();
    const float radius = params.sphere_radius * distance(camera.position, center);
    const float3 oc = camera.position - center;
    const float b = dot(oc, rd);
    const float h2 = b * b - (dot(oc, oc) - radius * radius);
    if (h2 < 0.f) {
        return hit;
    }
    const float t_in = max(-b - sqrt(h2), 0.f);
    const float t_out = -b + sqrt(h2);
    if (t_out <= 0.f) {
        return hit;
    }
    to_map = dir_map_frame(
        path_debug_map_normal(gbuffer.get_first_hit_normal(path_debug_selected_pixel())),
        params.map_frame == PATH_DEBUG_FRAME_LOCAL);
    const float3 enter = normalize(camera.position + rd * t_in - center);
    if (params.sphere_scale == 0) {
        hit.valid = true;
        hit.t = t_in;
        hit.dir = enter;
        hit.normal = enter;
        return hit;
    }

    const float step = (t_out - t_in) / float(SPHERE_STEPS);
    float t_prev = t_in;
    for (uint i = 1; i <= SPHERE_STEPS; i++) {
        const float t = t_in + step * float(i);
        const float3 q = camera.position + rd * t - center;
        const float d = length(q);
        if (d > 0.f && d > radius * sphere_extent(q / d, to_map)) {
            t_prev = t;
            continue;
        }
        float lo = t_prev;
        float hi = t;
        for (uint r = 0; r < SPHERE_REFINES; r++) {
            const float mid = 0.5f * (lo + hi);
            const float3 m = camera.position + rd * mid - center;
            const float dm = length(m);
            if (dm > 0.f && dm > radius * sphere_extent(m / dm, to_map)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hit.valid = true;
        hit.t = hi;
        hit.dir = normalize(camera.position + rd * hi - center);
        // the surface r(dir) * dir from two neighbor directions
        const float3 t1 = normalize(abs(hit.dir.z) < 0.9f ? cross(hit.dir, float3(0, 0, 1))
                                                          : cross(hit.dir, float3(1, 0, 0)));
        const float3 t2 = cross(hit.dir, t1);
        const float eps = 0.02f;
        const float3 d1 = normalize(hit.dir + eps * t1);
        const float3 d2 = normalize(hit.dir + eps * t2);
        const float3 p0 = sphere_extent(hit.dir, to_map) * hit.dir;
        const float3 n = cross(sphere_extent(d1, to_map) * d1 - p0,
                               sphere_extent(d2, to_map) * d2 - p0);
        hit.normal = length(n) > 0.f ? normalize(n) * sign(dot(n, hit.dir) + 1e-12f) : hit.dir;
        return hit;
    }
    return hit;
}

float3 compose(const uint2 pixel, const float3 src) {
    const uint2 render_pixel = path_debug_render_pixel(pixel);
    float3 color = backdrop_color(pixel, src, render_pixel);

    if (params.backdrop == PATH_DEBUG_BACKDROP_SPLIT_REFERENCE) {
        if (pixel.x == split_x() || pixel.x + 1 == split_x()) {
            color = SPLIT_COLOR;
        }
    }

    const uint2 source = path_debug_source_pixel(render_pixel);
    if (params.pixel_stats_view != PATH_DEBUG_PIXEL_STAT_OFF && source.x != PATH_DEBUG_NO_PIXEL) {
        const PathDebugPixelMoments m = path_debug_pixel_moments(source);
        if (m.count > 1) {
            const float value = path_debug_pixel_statistic(m, params.pixel_stats_view);
            color = lerp(color, colormap_turbo(saturate(value * params.pixel_stats_scale)),
                         params.false_color_opacity);
        }
    }

    if (merian_path_debug_gbuffer && params.heat_mode != PATH_DEBUG_HEAT_OFF) {
        const float linear_z = gbuffer.get_first_hit_linear_z(render_pixel);
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
        const float3 rd =
            camera.get_pinhole_direction_unjittered(float2(pixel), path_debug_out_dim());
        float3x3 to_map;
        const SphereHit hit = trace_sphere(rd, to_map);
        if (hit.valid && hit.t < scene_distance(pixel)) {
            const float3 map_dir = mul(to_map, hit.dir);
            const float3 uvw = dir_map_uv(params.map_transform, map_dir);
            if (uvw.z != 0.f && all(pixel == cursor_display_pixel())) {
                state[PATH_DEBUG_STATE_SPHERE_PROBE] =
                    path_debug_map_bin(path_debug_map_cell(uvw.xy)) + 1;
            }
            const float3 sphere_color = path_debug_map_color_dir(params.sphere_channel, map_dir) *
                                        (0.35f + 0.65f * saturate(-dot(hit.normal, rd)));
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
            const float3 filtered = recorded_radiance(pixel, true);
            const float lum = yuv_luminance(filtered);
            out_filtered[pixel] = float4(filtered, lum * lum);
        }
        if (merian_path_debug_gbuffer && params.image_depth != 0) {
            const float linear_z = gbuffer.get_first_hit_linear_z(pixel);
            const float2 grad_z = float2(gbuffer.get_grad_z(pixel));
            voxel_depth[pixel.y * params.dim_x + pixel.x] =
                float2(linear_z < MAX_LINEAR_Z ? linear_z : PATH_DEBUG_ENV_DISTANCE,
                       abs(grad_z.x) + abs(grad_z.y));
        }
        if (merian_path_debug_gbuffer && params.pick_x != PATH_DEBUG_NO_PIXEL &&
            all(pixel == uint2(params.pick_x, params.pick_y))) {
            const float linear_z = gbuffer.get_first_hit_linear_z(pixel);
            const float3 surface = camera.position_from_linear_z(
                pixel, path_debug_dim(),
                linear_z < MAX_LINEAR_Z ? linear_z : PATH_DEBUG_ENV_DISTANCE);
            const float3 n = gbuffer.get_first_hit_normal(pixel);
            const float3 normal = all(isfinite(n)) ? n : float3(0.f);
            path_debug_store_float3(state, PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_SURFACE,
                                    surface);
            path_debug_store_float3(state, PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_NORMAL, normal);
            const Hit hit = gbuffer.get_hit(pixel);
            state[PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_HIT] =
                hit.is_valid() ? hit.get_geometry_id() : PATH_RECORD_GEOMETRY_ENVIRONMENT;
            state[PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_HIT + 1] = hit.primitive_id;
            state[PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_SURFACE_SEQ] = params.pick_seq;
        }
    }

    if (all(pixel < path_debug_out_dim())) {
        const float4 src = in_src.SampleLevel(out_uv(pixel), 0.f);
        out_out[pixel] = float4(compose(pixel, src.rgb), src.a);
    }
}
