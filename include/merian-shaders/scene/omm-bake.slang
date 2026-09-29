// Bakes opacity micromaps for alpha-tested geometry, following NVIDIA's Opacity Micro-Map SDK.

__exported import merian_shaders.scene.scene_data;
__exported import merian_shaders.shading.materials.material_system;
import merian_shaders.utils.textures;

#include "merian-shaders/scene/omm-bake.slangh"

using merian;

static const uint OMM_STATE_TRANSPARENT = 0;
static const uint OMM_STATE_OPAQUE = 1;
static const uint OMM_STATE_UNKNOWN_OPAQUE = 3;

static const uint OMM_STATE_BITS = 2;
static const uint OMM_STATES_PER_WORD = 32 / OMM_STATE_BITS;

static const uint OMM_GROUP_SIZE = 64;
static const uint OMM_WORDS_PER_GROUP = OMM_GROUP_SIZE / OMM_STATES_PER_WORD;

static const uint OMM_SEEN_OPAQUE = 1;
static const uint OMM_SEEN_TRANSPARENT = 2;
static const uint OMM_SEEN_BOTH = OMM_SEEN_OPAQUE | OMM_SEEN_TRANSPARENT;

struct OmmBakeBinding {
    StructuredBuffer<OmmBakeJob, ScalarDataLayout> jobs;
    StructuredBuffer<OmmBakeBlock, ScalarDataLayout> blocks;
    uint block_count;
}

uint omm_scan(const MaterialSystem material_system,
              const MaterialID material_id,
              const float2 inv_texture_size,
              const OmmTexelFootprint footprint,
              const int first_row,
              const int row_step) {
    if (!footprint.bounded)
        return 0;

    const LOD0Sampler lod = LOD0Sampler();
    uint seen = 0;
    for (int y = footprint.first.y + first_row; y <= footprint.last.y; y += row_step) {
        bool was_inside = false;
        for (int x = footprint.first.x; x <= footprint.last.x; x++) {
            if (!omm_footprint_reads(footprint, x, y)) {
                if (was_inside)
                    break;
                continue;
            }
            was_inside = true;

            const float2 uv = (float2(float(x), float(y)) + 0.5f) * inv_texture_size;
            seen |= material_system.alpha_test(material_id, uv, lod) ? OMM_SEEN_TRANSPARENT
                                                                     : OMM_SEEN_OPAQUE;
            if (seen == OMM_SEEN_BOTH)
                return seen;
        }
    }
    return seen;
}

uint omm_state(const uint seen) {
    if (seen == OMM_SEEN_OPAQUE)
        return OMM_STATE_OPAQUE;
    if (seen == OMM_SEEN_TRANSPARENT)
        return OMM_STATE_TRANSPARENT;
    return OMM_STATE_UNKNOWN_OPAQUE;
}

groupshared uint omm_triangle_seen;
groupshared uint omm_words[OMM_WORDS_PER_GROUP];

[shader("compute")]
[numthreads(OMM_GROUP_SIZE, 1, 1)]
void main(uint3 group_id: SV_GroupID,
          uint3 local_id: SV_GroupThreadID,
          ParameterBlock<MaterialSystem> material_system,
          ParameterBlock<OmmBakeBinding> params) {
    if (group_id.x >= params.block_count)
        return;
    const uint thread = local_id.x;

    const OmmBakeBlock block = params.blocks[group_id.x];
    const OmmBakeJob job = params.jobs[block.job_index];
    const MaterialID material_id = job.geometry.material_id;
    const OmmDataAddress data = job.data + block.data_word_offset;

    const float2 texture_size = float2(material_system.get_alpha_texture_dimensions(material_id));
    const float2 inv_texture_size = 1.f / texture_size;
    float2 uv0;
    float2 uv1;
    float2 uv2;
    get_triangle_uv(job.geometry, block.primitive_id, uv0, uv1, uv2);
    const float2 t0 = uv0 * texture_size - 0.5f;
    const float2 dt1 = (uv1 - uv0) * texture_size;
    const float2 dt2 = (uv2 - uv0) * texture_size;

    if (thread == 0)
        omm_triangle_seen = 0;
    GroupMemoryBarrierWithGroupSync();
    const OmmTexelFootprint triangle = omm_texel_footprint(t0, t0 + dt1, t0 + dt2);
    uint dummy;
    InterlockedOr(omm_triangle_seen,
                  omm_scan(material_system, material_id, inv_texture_size, triangle, int(thread),
                           int(OMM_GROUP_SIZE)),
                  dummy);
    GroupMemoryBarrierWithGroupSync();
    const uint triangle_state = omm_state(omm_triangle_seen);

    const uint level = job.subdivision_level;
    const uint micro_triangles = 1u << (2 * level);
    const uint words = max(1u, micro_triangles / OMM_STATES_PER_WORD);

    if (triangle_state != OMM_STATE_UNKNOWN_OPAQUE) {
        uint packed = 0;
        for (uint i = 0; i < OMM_STATES_PER_WORD; i++) {
            packed |= triangle_state << (i * OMM_STATE_BITS);
        }
        for (uint word = thread; word < words; word += OMM_GROUP_SIZE) {
            data[word] = packed;
        }
        return;
    }

    for (uint base = 0; base < micro_triangles; base += OMM_GROUP_SIZE) {
        if (thread < OMM_WORDS_PER_GROUP)
            omm_words[thread] = 0;
        GroupMemoryBarrierWithGroupSync();

        const uint index = base + thread;
        if (index < micro_triangles) {
            const OmmMicroTriangle bary = omm_index_to_bary(index, level);
            const OmmTexelFootprint footprint =
                omm_texel_footprint(t0 + dt1 * bary.p0.x + dt2 * bary.p0.y,
                                    t0 + dt1 * bary.p1.x + dt2 * bary.p1.y,
                                    t0 + dt1 * bary.p2.x + dt2 * bary.p2.y);
            const uint state = omm_state(
                omm_scan(material_system, material_id, inv_texture_size, footprint, 0, 1));
            InterlockedOr(omm_words[thread / OMM_STATES_PER_WORD],
                          state << ((thread % OMM_STATES_PER_WORD) * OMM_STATE_BITS), dummy);
        }
        GroupMemoryBarrierWithGroupSync();

        const uint word = base / OMM_STATES_PER_WORD + thread;
        if (thread < OMM_WORDS_PER_GROUP && word < words)
            data[word] = omm_words[thread];
    }
}
