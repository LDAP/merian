// Bakes opacity micromaps for alpha-tested geometry.
//
// One thread per output word classifies the micro-triangles that share the word by evaluating the
// material's alpha test over each one; a micro-triangle only becomes opaque or transparent where
// every sample agrees, everything else stays unknown and still reaches the alpha test.
//
// The sampling follows the alpha texture's texel density, up to a budget per micro-triangle edge.
// Where a micro-triangle covers more texels than that, the cutout can vary between samples and be
// classified wrongly, which shows as a difference along cutout silhouettes; raising the
// subdivision level shrinks the micro-triangles and with them the error.

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

static const uint OMM_BAKE_GROUP_SIZE = 64;

struct OmmBakeBinding {
    StructuredBuffer<OmmBakeJob, ScalarDataLayout> jobs;
    RWStructuredBuffer<uint> data;
    RWStructuredBuffer<OmmTriangle, ScalarDataLayout> triangles;
    uint job_count;
    uint word_count;
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

// The three corner barycentrics of the micro-triangle a curve index addresses. Inverse of the
// barycentric-to-index mapping the Vulkan specification gives for opacity micromaps ("Ray Opacity
// Micromap"): the curve is defined so that both directions are bit-parallel.
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

// Opaque or transparent only where every sample of the micro-triangle agrees.
uint omm_classify(const MaterialSystem material_system,
                  const MaterialID material_id,
                  const float2 triangle_uv0,
                  const float2 duv1,
                  const float2 duv2,
                  const float2 b0,
                  const float2 b1,
                  const float2 b2,
                  const uint samples_per_edge) {
    const LOD0Sampler lod = LOD0Sampler();

    bool any_opaque = false;
    bool any_transparent = false;
    for (uint i = 0; i <= samples_per_edge; i++) {
        for (uint j = 0; i + j <= samples_per_edge; j++) {
            const float wi = float(i) / float(samples_per_edge);
            const float wj = float(j) / float(samples_per_edge);
            const float2 bary = b0 + (b1 - b0) * wi + (b2 - b0) * wj;
            const float2 uv = triangle_uv0 + duv1 * bary.x + duv2 * bary.y;

            if (material_system.alpha_test(material_id, uv, lod)) {
                any_transparent = true;
            } else {
                any_opaque = true;
            }
            if (any_opaque && any_transparent)
                return OMM_STATE_UNKNOWN_OPAQUE;
        }
    }

    return any_opaque ? OMM_STATE_OPAQUE : OMM_STATE_TRANSPARENT;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread_id: SV_DispatchThreadID,
          ParameterBlock<MaterialSystem> material_system,
          ParameterBlock<OmmBakeBinding> params) {
    const uint word = thread_id.x;
    if (word >= params.word_count)
        return;

    // the dispatch is flat over the words of every job
    uint job_index = 0;
    while (job_index + 1 < params.job_count && word >= params.jobs[job_index + 1].word_offset)
        job_index++;
    const OmmBakeJob job = params.jobs[job_index];

    const uint micro_triangles = 1u << (2 * job.subdivision_level);
    const uint words_per_triangle = max(1u, micro_triangles / OMM_STATES_PER_WORD);
    const uint job_word = word - job.word_offset;
    const uint primitive_id = job_word / words_per_triangle;
    const uint triangle_word = job_word - primitive_id * words_per_triangle;

    float2 uv_a;
    float2 uv_b;
    float2 uv_c;
    get_triangle_uv(job.geometry, primitive_id, uv_a, uv_b, uv_c);
    const float2 duv1 = uv_b - uv_a;
    const float2 duv2 = uv_c - uv_a;

    // one sample per texel along a micro-triangle edge, within the budget
    const float scale = 1.f / float(1u << job.subdivision_level);
    const float2 texels = float2(material_system.get_alpha_texture_dimensions(
        job.geometry.material_id));
    const float edge = max(max(length(duv1 * texels), length(duv2 * texels)),
                           length((duv2 - duv1) * texels)) * scale;
    const uint samples_per_edge =
        clamp(uint(ceil(edge)), 1u, max(1u, job.max_samples_per_edge));

    const uint first_state = triangle_word * OMM_STATES_PER_WORD;
    const uint state_count = min(OMM_STATES_PER_WORD, micro_triangles - first_state);

    uint packed = 0;
    for (uint i = 0; i < state_count; i++) {
        float2 uv0;
        float2 uv1;
        float2 uv2;
        omm_index_to_bary(first_state + i, job.subdivision_level, uv0, uv1, uv2);
        packed |= omm_classify(material_system, job.geometry.material_id, uv_a, duv1, duv2, uv0,
                               uv1, uv2, samples_per_edge)
                  << (i * OMM_STATE_BITS);
    }
    params.data[job.data_word_offset + job_word] = packed;

    if (triangle_word == 0) {
        OmmTriangle triangle;
        // relative to where this job's micromap data begins
        triangle.data_offset = 4 * primitive_id * words_per_triangle;
        triangle.subdivision_level = uint16_t(job.subdivision_level);
        triangle.format = uint16_t(MERIAN_OMM_FORMAT_4_STATE);
        params.triangles[job.triangle_offset + primitive_id] = triangle;
    }
}

