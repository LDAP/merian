// Bakes opacity micromaps for alpha-tested geometry.
//
// The alpha test runs at every texel a micro-triangle can read, and the state is only committed to
// opaque or transparent where they all agree; traversal then settles the hit the way the alpha
// test would have and the image does not change.
//
// Rasterizing a micro-triangle rather than sampling it is what keeps this affordable: its cost is
// the texel footprint it covers, not a fixed sample count, and it is what makes the answer exact.
// The approach is that of NVIDIA's Opacity Micro-Map SDK; the conservative edge test
// is Akenine-Moeller and Aila 2005, "Conservative and Tiled Rasterization Using a Modified
// Triangle Setup". The curve is the one the Vulkan specification defines under "Ray Opacity
// Micromap"; `tests/test_omm_bake.cpp` holds the same functions and checks them against brute
// force.

__exported import merian_shaders.scene.scene_data;
__exported import merian_shaders.shading.materials.material_system;
import merian_shaders.utils.textures;

#include "merian-shaders/scene/omm-bake.slangh"

using merian;

// Micro-triangle states, as the opacity micromap lookup interprets them.
static const uint OMM_STATE_TRANSPARENT = 0;
static const uint OMM_STATE_OPAQUE = 1;
static const uint OMM_STATE_UNKNOWN_OPAQUE = 3;

// 4-state format: two bits per micro-triangle, packed from the least significant bit up.
static const uint OMM_STATE_BITS = 2;
static const uint OMM_STATES_PER_WORD = 32 / OMM_STATE_BITS;

// Backstop for a uv mapping the chosen subdivision level does not tame - a degenerate or
// non-finite one. A micro-triangle reaching further than this stays unknown, which costs nothing
// but the alpha test that would have run anyway. The host picks the level so this rarely binds.
static const uint OMM_MAX_TEXELS = 4096;
static const int OMM_MAX_SPAN = 64;

static const uint OMM_GROUP_SIZE = 64;

struct OmmBakeBinding {
    StructuredBuffer<OmmBakeJob, ScalarDataLayout> jobs;
    StructuredBuffer<OmmBakeTriangle, ScalarDataLayout> triangles;
    // the micromap data of the one mesh this dispatch covers
    RWStructuredBuffer<uint> data;
    uint first_triangle;
    uint triangle_count;
}

uint omm_extract_even_bits(uint x) {
    x &= 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0f0f0f0f;
    x = (x | (x >> 4)) & 0x00ff00ff;
    x = (x | (x >> 8)) & 0x0000ffff;
    return x;
}

uint omm_prefix_eor(uint x) {
    x ^= x >> 1;
    x ^= x >> 2;
    x ^= x >> 4;
    x ^= x >> 8;
    return x;
}

// The barycentric corners of the micro-triangle a curve index addresses.
void omm_index_to_bary(const uint index,
                       const uint level,
                       out float2 uv0,
                       out float2 uv1,
                       out float2 uv2) {
    if (level == 0) {
        uv0 = float2(0, 0);
        uv1 = float2(1, 0);
        uv2 = float2(0, 1);
        return;
    }

    const uint b0 = omm_extract_even_bits(index);
    const uint b1 = omm_extract_even_bits(index >> 1);
    const uint fx = omm_prefix_eor(b0);
    const uint fy = omm_prefix_eor(b0 & ~b1);
    const uint t = fy ^ b1;

    const uint mask = (1u << level) - 1;
    uint iu = ((fx & ~t) | (b0 & ~t) | (~b0 & ~fx & t)) & mask;
    uint iv = (fy ^ b0) & mask;
    const uint iw = ((~fx & ~t) | (b0 & ~t) | (~b0 & fx & t)) & mask;

    // an inverted micro-triangle is the upright one of the next cell, mirrored
    const bool upright = bool((iu & 1) ^ (iv & 1) ^ (iw & 1));
    if (!upright) {
        iu += 1;
        iv += 1;
    }

    const float scale = 1.f / float(1u << level);
    const float step = upright ? scale : -scale;
    const float u = float(iu) * scale;
    const float v = float(iv) * scale;

    uv0 = float2(u, v);
    uv1 = float2(u + step, v);
    uv2 = float2(u, v + step);
}

// Signed area of the triangle, twice. Negative for the other winding.
float omm_edge_area(const float2 a, const float2 b, const float2 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

// Whether the axis-aligned box at `corner` with extent `extent` touches the half plane left of the
// edge a -> b.
bool omm_box_inside_edge(const float2 a, const float2 b, const float2 corner, const float2 extent) {
    const float2 normal = float2(b.y - a.y, a.x - b.x);
    const float offset = -dot(normal, a);
    const float2 bias = float2(normal.x > 0.f ? 0.f : normal.x, normal.y > 0.f ? 0.f : normal.y);
    return dot(normal, corner) + offset + dot(bias, extent) < 0.f;
}

// The corners of a triangle in texel space, where a texel centre sits on an integer.
void omm_triangle_texel_space(const GeometryData geometry,
                              const uint2 texture_size,
                              const uint primitive_id,
                              out float2 t0,
                              out float2 t1,
                              out float2 t2) {
    float2 uv0;
    float2 uv1;
    float2 uv2;
    get_triangle_uv(geometry, primitive_id, uv0, uv1, uv2);
    const float2 size = float2(texture_size);
    t0 = uv0 * size - 0.5f;
    t1 = uv1 * size - 0.5f;
    t2 = uv2 * size - 0.5f;
}

// The state of one micro-triangle, from every texel the alpha test can read inside it.
//
// A bilinear lookup reads the four lattice points around its position, so the texels to visit are
// the lattice points whose two-by-two neighbourhood the micro-triangle touches.
uint omm_classify(const MaterialSystem material_system,
                  const MaterialID material_id,
                  const uint2 texture_size,
                  const float2 p0,
                  const float2 p1,
                  const float2 p2) {
    const float2 aabb_min = min(p0, min(p1, p2));
    const float2 aabb_max = max(p0, max(p1, p2));
    if (!all(isfinite(aabb_min)) || !all(isfinite(aabb_max)))
        return OMM_STATE_UNKNOWN_OPAQUE;

    const int2 first = int2(floor(aabb_min));
    const int2 last = int2(floor(aabb_max)) + 1;
    const int2 span = max(int2(0), last - first) + 1;
    if (span.x > OMM_MAX_SPAN || span.y > OMM_MAX_SPAN)
        return OMM_STATE_UNKNOWN_OPAQUE;

    // the edge tests assume counter-clockwise corners
    const bool flip = omm_edge_area(p0, p1, p2) > 0.f;
    const float2 a = p0;
    const float2 b = flip ? p2 : p1;
    const float2 c = flip ? p1 : p2;

    const LOD0Sampler lod = LOD0Sampler();
    const float2 inv_size = 1.f / float2(texture_size);
    const float2 extent = float2(2, 2);

    bool any_opaque = false;
    bool any_transparent = false;
    uint visited = 0;
    for (int y = first.y; y <= last.y && visited < OMM_MAX_TEXELS; y++) {
        bool was_inside = false;
        for (int x = first.x; x <= last.x && visited < OMM_MAX_TEXELS; x++) {
            visited++;
            // the lattice point reads into the box one texel around it
            const float2 corner = float2(float(x) - 1.f, float(y) - 1.f);
            if (!omm_box_inside_edge(a, b, corner, extent) ||
                !omm_box_inside_edge(b, c, corner, extent) ||
                !omm_box_inside_edge(c, a, corner, extent)) {
                if (was_inside)
                    break;
                continue;
            }
            was_inside = true;

            // the same lookup the alpha test does, at the texel centre
            const float2 uv = (float2(float(x), float(y)) + 0.5f) * inv_size;
            if (material_system.alpha_test(material_id, uv, lod)) {
                any_transparent = true;
            } else {
                any_opaque = true;
            }
            if (any_opaque && any_transparent)
                return OMM_STATE_UNKNOWN_OPAQUE;
        }
    }

    if (any_opaque)
        return OMM_STATE_OPAQUE;
    if (any_transparent)
        return OMM_STATE_TRANSPARENT;
    // degenerate in texture space: no texel to go by
    return OMM_STATE_UNKNOWN_OPAQUE;
}

// One workgroup per triangle, walking the words of its own subdivision level.
[shader("compute")]
[numthreads(OMM_GROUP_SIZE, 1, 1)]
void main(uint3 group_id: SV_GroupID,
          uint3 local_id: SV_GroupThreadID,
          ParameterBlock<MaterialSystem> material_system,
          ParameterBlock<OmmBakeBinding> params) {
    if (group_id.x >= params.triangle_count)
        return;

    const OmmBakeTriangle triangle = params.triangles[params.first_triangle + group_id.x];
    const OmmBakeJob job = params.jobs[triangle.job_index];

    const uint2 texture_size =
        material_system.get_alpha_texture_dimensions(job.geometry.material_id);

    float2 t_a;
    float2 t_b;
    float2 t_c;
    omm_triangle_texel_space(job.geometry, texture_size, triangle.primitive_id, t_a, t_b, t_c);
    const float2 dt1 = t_b - t_a;
    const float2 dt2 = t_c - t_a;

    const uint micro_triangles = 1u << (2 * triangle.subdivision_level);
    const uint words = max(1u, micro_triangles / OMM_STATES_PER_WORD);

    for (uint word = local_id.x; word < words; word += OMM_GROUP_SIZE) {
        const uint first_state = word * OMM_STATES_PER_WORD;
        const uint state_count = min(OMM_STATES_PER_WORD, micro_triangles - first_state);

        uint packed = 0;
        for (uint i = 0; i < state_count; i++) {
            float2 b0;
            float2 b1;
            float2 b2;
            omm_index_to_bary(first_state + i, triangle.subdivision_level, b0, b1, b2);

            packed |= omm_classify(material_system, job.geometry.material_id, texture_size,
                                   t_a + dt1 * b0.x + dt2 * b0.y, t_a + dt1 * b1.x + dt2 * b1.y,
                                   t_a + dt1 * b2.x + dt2 * b2.y)
                      << (i * OMM_STATE_BITS);
        }
        params.data[triangle.data_word_offset + word] = packed;
    }
}
