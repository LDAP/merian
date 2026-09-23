import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.colors.colormaps;
import merian_shaders.colors.debug;
import merian_shaders.gbuffer;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

ParameterBlock<GBuffer> gbuffer;

static const float DEPTH_TEST_PIXELS = 2.f;
static const float DASH_PIXELS = 5.f;
static const float MAX_HALF_WIDTH = 6.f;
static const float QUAD_MARGIN = 2.f;
static const float MARKER_PADDING = 1.5f;

static const uint KIND_LINE = 0;
static const uint KIND_MARKER = 1;

struct Varyings {
    float4 position : SV_Position;
    nointerpolation float2 seg_a;
    nointerpolation float2 seg_b;
    nointerpolation float3 color;
    nointerpolation float z_ref;
    nointerpolation uint kind;
    float3 world_pos;
};

float half_width_at(const float z_ref, const float line_z) {
    const float width = params.overlay_thickness * z_ref / max(line_z, 1e-6f);
    return clamp(0.5f * width, 0.5f, MAX_HALF_WIDTH);
}

float4 clip_from_pixel(const float2 pixel, const float w) {
    const float2 ndc = (pixel / float2(path_debug_dim())) * 2.f - 1.f;
    return float4(ndc * w, 0.f, w);
}

float3 unescape(const float3 from, const float3 to) {
    const float3 d = to - from;
    const float len = length(d);
    if (len <= PATH_DEBUG_ENV_DISTANCE) {
        return to;
    }
    return from + d * (2.f * distance(from, camera.position) / len);
}

float view_depth(const float3 pos) {
    return dot(pos - camera.position, camera.W) / dot(camera.W, camera.W);
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

float3 method_color(const uint meta) {
    const uint method = min((meta >> PATH_RECORD_METHOD_SHIFT) & PATH_RECORD_METHOD_MASK,
                            PATH_RECORD_METHOD_COUNT - 1);
    const uint rgb = PATH_DEBUG_METHOD_COLORS[method];
    return float3((rgb >> 16) & 0xFFu, (rgb >> 8) & 0xFFu, rgb & 0xFFu) / 255.f;
}

Varyings segment(PathRecordStream records,
                 const uint vertex_region,
                 const uint path_index,
                 const uint kind,
                 const uint item,
                 const uint vertex_id,
                 Varyings o) {
    const PathRecordPath p = path_record_load_path(records, path_index);

    const float3 first = path_record_load_vertex(records, vertex_region, p.vertex_offset).pos;
    o.z_ref = max(distance(first, camera.position), 1e-6f);
    o.kind = kind;

    const uint end = kind == KIND_LINE ? item + 1 : item;
    if (end >= p.vertex_count) {
        return o;
    }
    const PathRecordVertex end_vertex =
        path_record_load_vertex(records, vertex_region, p.vertex_offset + end);
    const uint end_meta =
        path_record_load_vertex_meta(records, vertex_region, p.vertex_offset + end);
    if (path_debug_is_empty_draw(end_vertex)) {
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
        if (kind == KIND_MARKER) {
            p1 = p0;
        }
    }

    const float w_eps = camera.near / length(camera.W);
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

    // project_pinhole answers in the jittered pixel grid; SV_Position samples pixel centres
    const float2 a = camera.project_pinhole(p0, path_debug_dim()) - camera.jitter + 0.5f;
    const float2 b = camera.project_pinhole(p1, path_debug_dim()) - camera.jitter + 0.5f;
    o.seg_a = a;
    o.seg_b = b;

    if (kind == KIND_LINE) {
        float t;
        switch (params.overlay_color) {
        case PATH_DEBUG_COLOR_BOUNCES:
            t = float(path_debug_scatter_count(records, p, vertex_region)) / 16.f;
            break;
        case PATH_DEBUG_COLOR_RANDOM:
            t = 0.f;
            break;
        default:
            t = saturate((log2(max(p.luminance * params.overlay_color_scale, 1e-6f)) + 8.f) / 16.f);
            break;
        }
        o.color = params.overlay_color == PATH_DEBUG_COLOR_RANDOM ? random_color(path_index)
                                                                  : colormap_turbo(t);
        if (params.overlay_color == PATH_DEBUG_COLOR_TECHNIQUE) {
            o.color = method_color(path_debug_is_connection(end_meta)
                                       ? end_meta
                                       : path_record_load_vertex_meta(records, vertex_region,
                                                                      p.vertex_offset + parent));
        }
    } else {
        o.color = end == 0                                      ? float3(0.2f, 1.f, 0.2f)
                  : path_debug_is_connection(end_meta)          ? method_color(end_meta)
                  : escaped                                     ? float3(0.3f, 0.6f, 1.f)
                  : (end_meta & PATH_RECORD_FLAG_TERMINAL) != 0 ? float3(1.f, 0.25f, 0.25f)
                                                                : float3(1.f);
    }

    const uint2 corner = uint2(vertex_id == 1 || vertex_id == 3 || vertex_id == 4,
                               vertex_id == 2 || vertex_id == 4 || vertex_id == 5);
    const bool at_b = corner.x != 0;
    const float3 base_world = at_b ? p1 : p0;
    const float2 base = at_b ? b : a;
    const float w = at_b ? w1 : w0;

    const float2 ab = b - a;
    const float2 along = length(ab) > 1e-6f ? normalize(ab) : float2(1.f, 0.f);
    const float2 across = float2(-along.y, along.x);
    float half_width = half_width_at(o.z_ref, distance(base_world, camera.position)) + QUAD_MARGIN;
    if (kind == KIND_MARKER) {
        half_width += MARKER_PADDING;
    }

    const float2 cap =
        kind == KIND_MARKER ? along * (at_b ? half_width : -half_width) : float2(0.f);
    const float2 pixel = base + across * (corner.y != 0 ? half_width : -half_width) + cap;
    o.position = clip_from_pixel(pixel, w);
    o.world_pos = base_world;
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

    const uint per_kind = params.overlay_max_paths * PATH_DEBUG_MAX_SEGMENTS;
    const uint kind = instance_id >= per_kind ? KIND_MARKER : KIND_LINE;
    const uint index = instance_id - (kind == KIND_MARKER ? per_kind : 0);
    const uint slot = index / PATH_DEBUG_MAX_SEGMENTS;
    const uint item = index % PATH_DEBUG_MAX_SEGMENTS;
    const bool resampled_slots = params.overlay_rank == PATH_DEBUG_RANK_RESAMPLED;

    const uint path_index = resampled_slots ? slot : state[PATH_DEBUG_STATE_DRAW_LIST + slot];
    if ((params.overlay_isolate != PATH_DEBUG_NO_ISOLATE && path_index != params.overlay_isolate) ||
        slot >= params.overlay_max_paths) {
        return o;
    }
    if (resampled_slots) {
        if (resampled[PATH_DEBUG_RESAMPLED_KEYS + slot] == 0) {
            return o;
        }
        return segment(resampled, PATH_DEBUG_RESAMPLED_VERTICES, path_index, kind, item, vertex_id,
                       o);
    }
    if (slot >= state[PATH_DEBUG_STATE_DRAW_COUNT]) {
        return o;
    }
    return segment(in_records, path_record_vertex_region_offset(in_records), path_index, kind, item,
                   vertex_id, o);
}

[shader("fragment")]
float4 fragment_main(Varyings v): SV_Target {
    const float2 pixel = v.position.xy;
    const float2 ab = v.seg_b - v.seg_a;
    const float len2 = dot(ab, ab);
    const float t = len2 > 0.f ? clamp(dot(pixel - v.seg_a, ab) / len2, 0.f, 1.f) : 0.f;
    const float distance_to_line = distance(pixel, v.seg_a + t * ab);

    const float line_z = distance(v.world_pos, camera.position);
    float half_width = half_width_at(v.z_ref, line_z);
    if (v.kind == KIND_MARKER) {
        half_width += MARKER_PADDING;
    }
    const float coverage = saturate(half_width + 0.5f - distance_to_line);
    if (coverage <= 0.f) {
        discard;
    }

    if (merian_path_debug_gbuffer && params.overlay_depth_mode != PATH_DEBUG_DEPTH_XRAY) {
        const uint2 p = uint2(pixel);
        const float gz = gbuffer.get_linear_z(p);
        const float2 grad = float2(gbuffer.get_grad_z(p));
        const float tolerance = DEPTH_TEST_PIXELS * (abs(grad.x) + abs(grad.y)) + 0.01f * gz;
        if (gz > 0.f && gz < PATH_DEBUG_ENV_DISTANCE && line_z > gz + tolerance) {
            if (params.overlay_depth_mode != PATH_DEBUG_DEPTH_DASH) {
                discard;
            }
            if ((uint(t * sqrt(len2) / DASH_PIXELS) & 1u) != 0u) {
                discard;
            }
        }
    }

    const float rim = saturate((distance_to_line / half_width - 0.7f) / 0.3f);
    return float4(v.color * (1.f - 0.5f * rim), params.overlay_alpha * coverage);
}
