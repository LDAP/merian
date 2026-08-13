import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.colors.colormaps;
import merian_shaders.colors.debug;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

// covers a full-diagonal segment at 4K; longer ones fall back to a coarser stride
static const uint MAX_DDA_STEPS = 8192;
static const float ENV_SEGMENT_LENGTH = 1e6f;

void splat(const int2 p, const uint packed) {
    if (all(p >= 0) && all(p < int2(path_debug_dim()))) {
        InterlockedMax(overlay[p.y * params.dim_x + p.x], packed);
    }
}

void splat_marker(const float2 p, const float3 color) {
    const uint packed = path_debug_pack_overlay(255, color);
    const int2 c = int2(round(p));
    splat(c, packed);
    splat(c + int2(1, 0), packed);
    splat(c + int2(-1, 0), packed);
    splat(c + int2(0, 1), packed);
    splat(c + int2(0, -1), packed);
}

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint slot = gid.x / PATH_DEBUG_MAX_SEGMENTS;
    const uint seg = gid.x % PATH_DEBUG_MAX_SEGMENTS;
    if (slot >= min(state[PATH_DEBUG_STATE_DRAW_COUNT], params.max_draw)) {
        return;
    }

    const uint path_index = state[PATH_DEBUG_STATE_DRAW_LIST + slot];
    if (params.focus_path != PATH_DEBUG_NO_FOCUS && path_index != params.focus_path) {
        return;
    }
    const PathRecordPath p = path_record_load_path(in_records, path_index);
    if (seg + 1 >= p.vertex_count) {
        return;
    }

    const uint vro = path_record_vertex_region_offset(in_records);
    const PathRecordVertex v0 = path_record_load_vertex(in_records, vro, p.vertex_offset + seg);
    PathRecordVertex v1 = path_record_load_vertex(in_records, vro, p.vertex_offset + seg + 1);

    // pull environment escapes to a drawable distance
    const float3 d = v1.pos - v0.pos;
    const float seg_len = length(d);
    const bool escaped = seg_len > ENV_SEGMENT_LENGTH;
    if (escaped) {
        v1.pos = v0.pos + d * (2.f * distance(v0.pos, camera.position) / seg_len);
    }

    // clip against the near plane, in units of the focal distance along W
    const float inv_w2 = 1.f / dot(camera.W, camera.W);
    const float w_eps = camera.near * sqrt(inv_w2);
    float w0 = dot(v0.pos - camera.position, camera.W) * inv_w2;
    float w1 = dot(v1.pos - camera.position, camera.W) * inv_w2;
    if (w0 < w_eps && w1 < w_eps) {
        return;
    }
    float3 p0 = v0.pos;
    float3 p1 = v1.pos;
    if (w0 < w_eps) {
        p0 = lerp(v0.pos, v1.pos, (w_eps - w0) / (w1 - w0));
    } else if (w1 < w_eps) {
        p1 = lerp(v0.pos, v1.pos, (w_eps - w1) / (w0 - w1));
    }

    float2 a = camera.project_pinhole(p0, path_debug_dim());
    float2 b = camera.project_pinhole(p1, path_debug_dim());

    // clip to the viewport so DDA steps are spent on visible pixels
    float t0 = 0.f;
    float t1 = 1.f;
    const float2 ab = b - a;
    for (int axis = 0; axis < 2; axis++) {
        for (int side = 0; side < 2; side++) {
            const float bound = side == 0 ? 0.f : float(path_debug_dim()[axis] - 1);
            const float dist = side == 0 ? bound - a[axis] : a[axis] - bound;
            const float delta = side == 0 ? ab[axis] : -ab[axis];
            if (abs(delta) < 1e-12f) {
                if (dist > 0.f) {
                    return;
                }
            } else if (delta > 0.f) {
                t0 = max(t0, dist / delta);
            } else {
                t1 = min(t1, dist / delta);
            }
        }
    }
    if (t0 > t1) {
        return;
    }

    float t;
    switch (params.color_mode) {
    case PATH_DEBUG_COLOR_BOUNCES:
        t = float(p.vertex_count) / 16.f;
        break;
    case PATH_DEBUG_COLOR_RANDOM:
        t = 0.f;
        break;
    default:
        t = saturate((log2(max(p.luminance * params.color_scale, 1e-6f)) + 8.f) / 16.f);
        break;
    }
    const float3 color = params.color_mode == PATH_DEBUG_COLOR_RANDOM ? random_color(path_index)
                                                                       : colormap_turbo(t);
    const uint packed = path_debug_pack_overlay(1 + uint(t * 253.f), color);

    const float2 ca = a + t0 * ab;
    const float2 cb = a + t1 * ab;
    const uint steps = clamp(uint(ceil(max(abs(cb.x - ca.x), abs(cb.y - ca.y)))), 1, MAX_DDA_STEPS);
    for (uint s = 0; s <= steps; s++) {
        const float2 pos = lerp(ca, cb, float(s) / float(steps));
        splat(int2(round(pos)), packed);
    }

    // vertex markers: first = green, terminal = red, escaped = blue, interior = white
    if (seg == 0 && w0 >= w_eps) {
        splat_marker(a, float3(0.2f, 1.f, 0.2f));
    }
    if (w1 >= w_eps) {
        float3 marker = float3(1.f);
        if (escaped) {
            marker = float3(0.3f, 0.6f, 1.f);
        } else if (v1.terminal) {
            marker = float3(1.f, 0.25f, 0.25f);
        }
        splat_marker(b, marker);
    }
}
