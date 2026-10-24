import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.frame;
import merian_shaders.colors.colormaps;
import merian_shaders.colors.colorspaces;
import merian_shaders.colors.debug;
import merian_shaders.gbuffer;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"
#include "merian-shaders/utils/math-constants.slangh"

ParameterBlock<GBuffer> gbuffer;
[vk::push_constant] ConstantBuffer<PathDebugOverlayView> view;

// in pixels at a display scale of 1
static const float DEPTH_TEST_PIXELS = 2.f;
static const float DASH_PIXELS = 5.f;
static const float MAX_HALF_WIDTH = 6.f;
static const float QUAD_MARGIN = 2.f;
static const float MARKER_PADDING = 1.5f;
static const float RING_HALF_WIDTH = 1.f;
static const float POINT_ARM_INNER = 4.f;
static const float POINT_ARM_OUTER = 11.f;
static const float POINT_HALF_WIDTH = 1.f;
static const float POINT_OUTLINE = 1.25f;

static const float OCCLUDED_ALPHA = 0.4f;
static const float3 SELECTION_COLOR = float3(0.35f, 0.85f, 1.f);
static const float3 POINT_COLOR = float3(1.f, 0.95f, 0.15f);
static const float3 NO_LIGHT_COLOR = float3(0.25f);

static const uint KIND_LINE = 0;
static const uint KIND_VERTEX = 1;
static const uint KIND_RING = 2;
static const uint KIND_POINT = 3;

struct Varyings {
    float4 position : SV_Position;
    nointerpolation float2 seg_a;
    nointerpolation float2 seg_b;
    nointerpolation float3 color;
    nointerpolation float z_ref;
    nointerpolation uint kind;
    float3 world_pos;
};

float2 view_origin() {
    return float2(view.x, view.y);
}

float2 view_size() {
    return float2(view.w, view.h);
}

// the camera without its jitter: the displayed image is the converged one
float2 to_target(const float3 pos) {
    return view_origin() + (camera.project_pinhole(pos, uint2(1)) + 0.5f) * view_size();
}

float4 clip_from_target(const float2 pixel, const float w) {
    const float2 ndc = (pixel - view_origin()) / view_size() * 2.f - 1.f;
    return float4(ndc * w, 0.f, w);
}

float view_depth(const float3 pos) {
    return dot(pos - camera.position, camera.W) / dot(camera.W, camera.W);
}

float half_width_at(const uint kind, const float z_ref, const float line_z) {
    if (kind == KIND_RING) {
        return view.scale * RING_HALF_WIDTH;
    }
    const float width = params.overlay_thickness * z_ref / max(line_z, 1e-6f);
    const float half_width = clamp(0.5f * width, 0.5f, MAX_HALF_WIDTH) +
                             (kind == KIND_VERTEX ? MARKER_PADDING : 0.f);
    return view.scale * half_width;
}

// A surface lies in front of the distance at every record pixel around the target pixel, so the
// test does not follow the jitter of the gbuffer. Depths clamped to MAX_LINEAR_Z are the sky.
bool behind_geometry(const float2 target_pixel, const float distance_to_camera) {
    if (!merian_path_debug_gbuffer) {
        return false;
    }
    const int2 dim = int2(path_debug_dim());
    const float2 at = (target_pixel - view_origin()) / view_size() * float2(dim) - 0.5f;
    const int2 base = int2(floor(at));
    for (int i = 0; i < 4; i++) {
        const uint2 p = uint2(clamp(base + int2(i & 1, i >> 1), int2(0), dim - 1));
        const float gz = gbuffer.get_first_hit_linear_z(p);
        if (!(gz > 0.f && gz < MAX_LINEAR_Z)) {
            return false;
        }
        const float2 grad = float2(gbuffer.get_grad_z(p));
        const float tolerance = DEPTH_TEST_PIXELS * (abs(grad.x) + abs(grad.y)) + 0.01f * gz;
        if (distance_to_camera <= gz + tolerance) {
            return false;
        }
    }
    return true;
}

float3 unescape(const float3 from, const float3 to) {
    const float3 d = to - from;
    const float len = length(d);
    if (len <= PATH_DEBUG_ENV_DISTANCE) {
        return to;
    }
    return from + d * (2.f * distance(from, camera.position) / len);
}

uint chain_parent(PathRecordStream records,
                  const PathRecordPath p,
                  const uint vertex_region,
                  const uint vertex) {
    uint parent = vertex - 1;
    while (parent > 0 && path_debug_is_connection(path_record_load_vertex_meta(
                             records, vertex_region, p.vertex_offset + parent))) {
        parent--;
    }
    return parent;
}

float3 packed_color(const uint rgb) {
    return srgb_to_rgb(path_debug_unpack_rgb(rgb));
}

float3 method_color(const uint method) {
    return packed_color(PATH_DEBUG_METHOD_COLORS[min(method, PATH_RECORD_METHOD_COUNT - 1)]);
}

float finite_luminance(const float3 rgb) {
    return path_debug_finite_or_zero(yuv_luminance(rgb));
}

// The radiance a segment carries towards the camera: what the selection's events from its far end
// on contribute, over the path's weight up to its near end.
float3 carried_radiance(PathRecordStream records,
                        const PathRecordPath p,
                        const uint vertex_region,
                        const uint2 selected,
                        const uint end,
                        const uint parent) {
    const PathRecordVertex end_vertex =
        path_record_load_vertex(records, vertex_region, p.vertex_offset + end);
    // a connection's weight runs up to the light
    if (end_vertex.nee) {
        return select(end_vertex.throughput_rgb > 0.f,
                      path_debug_finite_or_zero(end_vertex.contrib) / end_vertex.throughput_rgb,
                      float3(0.f));
    }
    float3 carried = float3(0.f);
    for (uint v = end; v < p.vertex_count; v++) {
        if (path_debug_is_event_vertex(selected, v)) {
            carried += path_debug_finite_or_zero(
                path_record_load_vertex(records, vertex_region, p.vertex_offset + v).contrib);
        }
    }
    const float3 throughput =
        path_record_load_vertex(records, vertex_region, p.vertex_offset + parent).throughput_rgb;
    return select(throughput > 0.f, carried / throughput, float3(0.f));
}

// The weight the segment's direction was drawn with at the vertex it leaves, BSDF·cos/pdf: the
// throughput after the draw over the throughput before it, 1 at the camera.
float sample_weight(PathRecordStream records,
                    const PathRecordPath p,
                    const uint vertex_region,
                    const uint end,
                    const uint parent) {
    const PathRecordVertex end_vertex =
        path_record_load_vertex(records, vertex_region, p.vertex_offset + end);
    const float drawn =
        path_record_load_vertex(records, vertex_region,
                                p.vertex_offset + (end_vertex.nee ? end : parent))
            .throughput;
    const float before =
        parent == 0 ? 1.f
                    : path_record_load_vertex(records, vertex_region,
                                              p.vertex_offset +
                                                  chain_parent(records, p, vertex_region, parent))
                          .throughput;
    return before > 0.f ? drawn / before : 0.f;
}

float stops_coordinate(const float value, const float stops) {
    return saturate(log2(max(value, 1e-12f)) / stops + 0.5f);
}

float color_coordinate(const float value, const float stops) {
    return stops_coordinate(value * params.overlay_color_scale, stops);
}

float near_depth() {
    return camera.near / length(camera.W);
}

uint2 quad_corner(const uint vertex_id) {
    return uint2(vertex_id == 1 || vertex_id == 3 || vertex_id == 4,
                 vertex_id == 2 || vertex_id == 4 || vertex_id == 5);
}

Varyings emit_quad(float3 p0, float3 p1, const uint kind, const uint vertex_id, Varyings o) {
    const float w_eps = near_depth();
    float w0 = view_depth(p0);
    float w1 = view_depth(p1);
    if (w0 < w_eps && w1 < w_eps) {
        return o;
    }
    if (w0 < w_eps) {
        p0 = lerp(p0, p1, (w_eps - w0) / (w1 - w0));
        w0 = w_eps;
    } else if (w1 < w_eps) {
        p1 = lerp(p0, p1, (w0 - w_eps) / (w0 - w1));
        w1 = w_eps;
    }

    const float2 a = to_target(p0);
    const float2 b = to_target(p1);
    o.seg_a = a;
    o.seg_b = b;
    o.kind = kind;

    const uint2 corner = quad_corner(vertex_id);
    const bool at_b = corner.x != 0;
    const float3 base_world = at_b ? p1 : p0;
    const float2 base = at_b ? b : a;

    const float2 ab = b - a;
    const float2 along = length(ab) > 1e-6f ? normalize(ab) : float2(1.f, 0.f);
    const float2 across = float2(-along.y, along.x);
    const float half_width = half_width_at(kind, o.z_ref, distance(base_world, camera.position)) +
                             view.scale * QUAD_MARGIN;

    const float2 cap =
        kind == KIND_VERTEX ? along * (at_b ? half_width : -half_width) : float2(0.f);
    const float2 pixel = base + across * (corner.y != 0 ? half_width : -half_width) + cap;
    o.position = clip_from_target(pixel, at_b ? w1 : w0);
    o.world_pos = base_world;
    return o;
}

Varyings segment(PathRecordStream records,
                 const uint vertex_region,
                 const uint path_index,
                 const uint kind,
                 const uint item,
                 const uint vertex_id,
                 Varyings o) {
    const PathRecordPath p = path_record_load_path(records, path_index);
    const uint2 selected = path_debug_kept_events(path_index);

    const float3 first = path_record_load_vertex(records, vertex_region, p.vertex_offset).pos;
    o.z_ref = max(distance(first, camera.position), 1e-6f);

    const uint end = kind == KIND_LINE ? item + 1 : item;
    if (end >= p.vertex_count) {
        return o;
    }
    const PathRecordVertex end_vertex =
        path_record_load_vertex(records, vertex_region, p.vertex_offset + end);
    if (path_debug_is_empty_draw(end_vertex) ||
        !path_debug_event_drawn(selected, end, end_vertex.nee, path_debug_paths_filtered())) {
        return o;
    }
    const uint parent = end > 0 ? chain_parent(records, p, vertex_region, end) : 0;
    const float3 raw = end_vertex.pos;
    float3 p0 = raw;
    float3 p1 = raw;
    bool escaped = false;
    if (end > 0) {
        const float3 from =
            path_record_load_vertex(records, vertex_region, p.vertex_offset + parent).pos;
        p1 = unescape(from, raw);
        escaped = any(p1 != raw);
        p0 = kind == KIND_LINE ? from : p1;
    }

    if (kind == KIND_LINE) {
        switch (params.overlay_color) {
        case PATH_DEBUG_COLOR_CONTRIBUTION:
            o.color = srgb_to_rgb(colormap_plasma(color_coordinate(
                yuv_luminance(carried_radiance(records, p, vertex_region, selected, end, parent)),
                PATH_DEBUG_CONTRIBUTION_STOPS)));
            break;
        case PATH_DEBUG_COLOR_TRANSPORTED: {
            const float3 carried =
                carried_radiance(records, p, vertex_region, selected, end, parent);
            const float peak = max(carried.r, max(carried.g, carried.b));
            o.color = peak > 0.f ? carried / peak : NO_LIGHT_COLOR;
            break;
        }
        case PATH_DEBUG_COLOR_SCATTER_EVENTS:
            o.color = srgb_to_rgb(colormap_turbo(
                float(path_debug_events_depth(records, p, vertex_region, selected)) /
                float(PATH_DEBUG_SCATTER_COLOR_EVENTS)));
            break;
        case PATH_DEBUG_COLOR_RANDOM:
            o.color = random_color(path_index);
            break;
        case PATH_DEBUG_COLOR_SAMPLE_WEIGHT:
            o.color = srgb_to_rgb(colormap_diverging(
                stops_coordinate(sample_weight(records, p, vertex_region, end, parent),
                                 PATH_DEBUG_SAMPLE_WEIGHT_STOPS)));
            break;
        case PATH_DEBUG_COLOR_TECHNIQUE:
            o.color = method_color(
                end_vertex.nee
                    ? end_vertex.method
                    : path_record_load_vertex(records, vertex_region, p.vertex_offset + parent)
                          .method);
            break;
        default:
            o.color = srgb_to_rgb(
                colormap_turbo(color_coordinate(p.luminance, PATH_DEBUG_LUMINANCE_STOPS)));
            break;
        }
    } else {
        const bool last = int(end) == path_debug_last_event(selected);
        o.color = end == 0                       ? packed_color(PATH_DEBUG_START_COLOR)
                  : end_vertex.nee               ? method_color(end_vertex.method)
                  : escaped                      ? packed_color(PATH_DEBUG_ESCAPED_COLOR)
                  : end_vertex.terminal || last  ? packed_color(PATH_DEBUG_END_COLOR)
                                                 : float3(1.f);
    }
    return emit_quad(p0, p1, kind, vertex_id, o);
}

float3 ring_point(const float3 center, const float3x3 frame, const float r, const uint i) {
    const float a = 2.f * MERIAN_M_PI * float(i) / float(PATH_DEBUG_RING_SEGMENTS);
    return center + r * (cos(a) * frame[0] + sin(a) * frame[1]);
}

// The region's sphere where it meets the selected surface, a circle in its tangent plane.
Varyings ring_segment(const uint index, const uint vertex_id, Varyings o) {
    if (!path_debug_selects_region() || params.sel_target != PATH_DEBUG_TARGET_SPHERE ||
        params.sel_point_valid == 0) {
        return o;
    }
    const float3 center = path_debug_selected_point();
    const float3 normal = path_debug_selected_normal();
    const float3x3 frame =
        make_frame(any(normal != 0.f) ? normalize(normal) : normalize(camera.position - center));
    const float r = params.sel_region_radius;
    o.color = SELECTION_COLOR;
    o.z_ref = 1.f;
    return emit_quad(ring_point(center, frame, r, index), ring_point(center, frame, r, index + 1),
                     KIND_RING, vertex_id, o);
}

// A crosshair on the selected point, or on its pixel until the point is known.
Varyings point_marker(const uint vertex_id, Varyings o) {
    float2 center;
    if (params.sel_point_valid != 0) {
        const float3 point = path_debug_selected_point();
        if (view_depth(point) < near_depth()) {
            return o;
        }
        center = to_target(point);
        o.world_pos = point;
    } else if (path_debug_has_pixel()) {
        center = view_origin() + (float2(path_debug_selected_pixel()) + 0.5f) /
                                     float2(path_debug_dim()) * view_size();
    } else {
        return o;
    }
    const uint2 corner = quad_corner(vertex_id);
    const float half_size = view.scale * (POINT_ARM_OUTER + POINT_OUTLINE + QUAD_MARGIN);
    o.seg_a = center;
    o.kind = KIND_POINT;
    o.color = POINT_COLOR;
    o.position = clip_from_target(center + half_size * (float2(corner) * 2.f - 1.f), 1.f);
    return o;
}

[shader("vertex")]
Varyings vertex_main(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID) {
    Varyings o;
    o.position = float4(0.f, 0.f, 0.f, 1.f);
    o.seg_a = float2(0.f);
    o.seg_b = float2(0.f);
    o.color = float3(0.f);
    o.z_ref = 1.f;
    o.kind = KIND_LINE;
    o.world_pos = camera.position;

    const uint per_kind =
        path_debug_drawn_paths(params.overlay_max_paths, params.overlay_color) *
        PATH_DEBUG_MAX_SEGMENTS;
    if (instance_id >= 2 * per_kind) {
        const uint index = instance_id - 2 * per_kind;
        return index < PATH_DEBUG_RING_SEGMENTS ? ring_segment(index, vertex_id, o)
                                                : point_marker(vertex_id, o);
    }
    const uint kind = instance_id >= per_kind ? KIND_VERTEX : KIND_LINE;
    const uint index = instance_id - (kind == KIND_VERTEX ? per_kind : 0);
    const uint slot = index / PATH_DEBUG_MAX_SEGMENTS;
    const uint item = index % PATH_DEBUG_MAX_SEGMENTS;
    if (!path_debug_slot_kept(slot) ||
        (params.overlay_isolate != PATH_DEBUG_NO_PATH && slot != params.overlay_isolate)) {
        return o;
    }
    return segment(kept, PATH_DEBUG_KEPT_VERTICES, slot, kind, item, vertex_id, o);
}

float closest_on_segment(const float2 pixel, const float2 a, const float2 b) {
    const float2 ab = b - a;
    const float len2 = dot(ab, ab);
    return len2 > 0.f ? clamp(dot(pixel - a, ab) / len2, 0.f, 1.f) : 0.f;
}

float segment_distance(const float2 pixel, const float2 a, const float2 b) {
    return distance(pixel, lerp(a, b, closest_on_segment(pixel, a, b)));
}

float4 point_fragment(const Varyings v) {
    const float2 d = abs(v.position.xy - v.seg_a) / view.scale;
    const float arm = min(segment_distance(d, float2(POINT_ARM_INNER, 0.f),
                                           float2(POINT_ARM_OUTER, 0.f)),
                          segment_distance(d, float2(0.f, POINT_ARM_INNER),
                                           float2(0.f, POINT_ARM_OUTER)));
    const float fill = saturate(POINT_HALF_WIDTH + 0.5f - arm);
    const float outline = saturate(POINT_HALF_WIDTH + POINT_OUTLINE + 0.5f - arm);
    if (outline <= 0.f) {
        discard;
    }
    const bool hidden = params.sel_point_valid != 0 &&
                        behind_geometry(v.seg_a, distance(v.world_pos, camera.position));
    const float3 color = v.color * fill / max(outline, 1e-6f);
    return float4(color, outline * (hidden ? OCCLUDED_ALPHA : 1.f));
}

[shader("fragment")]
float4 fragment_main(Varyings v): SV_Target {
    if (v.kind == KIND_POINT) {
        return point_fragment(v);
    }
    const float2 pixel = v.position.xy;
    const float t = closest_on_segment(pixel, v.seg_a, v.seg_b);
    // tested where the line runs, so a wide line is hidden or dashed across its whole width
    const float2 on_line = lerp(v.seg_a, v.seg_b, t);
    const float distance_to_line = distance(pixel, on_line);

    const float line_z = distance(v.world_pos, camera.position);
    const float half_width = half_width_at(v.kind, v.z_ref, line_z);
    const float coverage = saturate(half_width + 0.5f - distance_to_line);
    if (coverage <= 0.f) {
        discard;
    }

    float alpha = params.overlay_alpha;
    if (v.kind == KIND_RING) {
        alpha = behind_geometry(on_line, line_z) ? OCCLUDED_ALPHA : 1.f;
    } else if (params.overlay_depth_mode != PATH_DEBUG_DEPTH_XRAY &&
               behind_geometry(on_line, line_z)) {
        if (params.overlay_depth_mode == PATH_DEBUG_DEPTH_HIDE) {
            discard;
        }
        if ((uint(distance(on_line, v.seg_a) / (DASH_PIXELS * view.scale)) & 1u) != 0u) {
            discard;
        }
    }

    const float rim = saturate((distance_to_line / half_width - 0.7f) / 0.3f);
    return float4(v.color * (1.f - 0.5f * rim), alpha * coverage);
}
