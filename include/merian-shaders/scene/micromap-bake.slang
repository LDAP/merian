// Bakes opacity micromaps for alpha-tested geometry, following NVIDIA's Opacity Micro-Map SDK.

__exported import merian_shaders.scene.scene_data;
__exported import merian_shaders.shading.materials.material_system;
import merian_shaders.colors.colorspaces;
import merian_shaders.utils.textures;

#include "merian-shaders/scene/micromap-bake.slangh"

using merian;

static const uint OMM_STATE_TRANSPARENT = 0;
static const uint OMM_STATE_OPAQUE = 1;
static const uint OMM_STATE_UNKNOWN_OPAQUE = 3;

static const uint OMM_STATE_BITS = 2;
static const uint OMM_STATES_PER_WORD = 32 / OMM_STATE_BITS;

static const uint BAKE_GROUP_SIZE = 64;
static const uint OMM_WORDS_PER_GROUP = BAKE_GROUP_SIZE / OMM_STATES_PER_WORD;

static const uint SEEN_OPAQUE = 1;
static const uint SEEN_TRANSPARENT = 2;
static const uint SEEN_EMITTING = 4;
static const uint SEEN_DARK = 8;

enum MicromapKind {
    Opacity,
    Cutout,
    Emission,
};

uint scan_settled(const MicromapKind kind) {
    switch (kind) {
    case MicromapKind::Opacity:
        return SEEN_OPAQUE | SEEN_TRANSPARENT;
    case MicromapKind::Cutout:
        return SEEN_OPAQUE;
    default:
        return SEEN_EMITTING | SEEN_DARK;
    }
}

struct MicromapBakeBinding {
    StructuredBuffer<MicromapBakeJob, ScalarDataLayout> jobs;
    StructuredBuffer<MicromapBakeBlock, ScalarDataLayout> blocks;
    uint block_count;
}

float lookup_reach(const bool nearest) {
    return nearest ? 0.5f : 1.f;
}

struct TexelGrid {
    float2 inv_size;
    float2 t0;
    float2 dt1;
    float2 dt2;
    float reach;

    __init(const float2 size,
           const float reach,
           const float2 uv0,
           const float2 uv1,
           const float2 uv2) {
        inv_size = 1.f / size;
        t0 = uv0 * size - 0.5f;
        dt1 = (uv1 - uv0) * size;
        dt2 = (uv2 - uv0) * size;
        this.reach = reach;
    }

    bool reads(const TexelFootprint footprint, const int x, const int y) {
        return micromap_footprint_reaches(footprint, x, y, reach);
    }

    TexelFootprint triangle_footprint() {
        return micromap_texel_footprint(t0, t0 + dt1, t0 + dt2);
    }

    TexelFootprint footprint(const MicroTriangle bary) {
        return micromap_texel_footprint(t0 + dt1 * bary.p0.x + dt2 * bary.p0.y,
                                        t0 + dt1 * bary.p1.x + dt2 * bary.p1.y,
                                        t0 + dt1 * bary.p2.x + dt2 * bary.p2.y);
    }

    float2 texel_center(const int x, const int y) {
        return (float2(float(x), float(y)) + 0.5f) * inv_size;
    }
}

float emitted_luminance(const MaterialSystem material_system,
                        const MaterialID material_id,
                        const float2 uv,
                        const ExplicitLODSampler lod) {
    const float3 n = float3(0, 0, 1);
    return yuv_luminance(material_system.get_emission(
        material_id,
        ShadingData(n, float3(0), uv, 0.f, OrthonormalFrame(n), n, uint16_t(1), true), lod));
}

uint scan(const MaterialSystem material_system,
          const MaterialID material_id,
          const MicromapKind kind,
          const TexelGrid grid,
          const ExplicitLODSampler lod,
          const TexelFootprint footprint,
          const int first_row,
          const int row_step) {
    const uint settled = scan_settled(kind);
    if (!footprint.bounded)
        return settled;

    uint seen = 0;
    for (int y = footprint.first.y + first_row; y <= footprint.last.y; y += row_step) {
        bool was_inside = false;
        for (int x = footprint.first.x; x <= footprint.last.x; x++) {
            if (!grid.reads(footprint, x, y)) {
                if (was_inside)
                    break;
                continue;
            }
            was_inside = true;

            const float2 uv = grid.texel_center(x, y);
            if (kind == MicromapKind::Emission) {
                seen |= emitted_luminance(material_system, material_id, uv, lod) > 0.f
                            ? SEEN_EMITTING
                            : SEEN_DARK;
            } else {
                seen |= material_system.alpha_test(material_id, uv, lod) ? SEEN_TRANSPARENT
                                                                         : SEEN_OPAQUE;
            }
            if ((seen & settled) == settled)
                return seen;
        }
    }
    return seen;
}

uint settle_emission(const uint seen) {
    return (seen & (SEEN_EMITTING | SEEN_DARK)) == 0 ? seen | SEEN_EMITTING : seen;
}

uint omm_state(const uint seen) {
    if (seen == SEEN_OPAQUE)
        return OMM_STATE_OPAQUE;
    if (seen == SEEN_TRANSPARENT)
        return OMM_STATE_TRANSPARENT;
    return OMM_STATE_UNKNOWN_OPAQUE;
}

groupshared uint opacity_seen;
groupshared uint emission_seen;
groupshared uint omm_words[OMM_WORDS_PER_GROUP];
groupshared uint emission_words[EMISSION_MAX_BIT_WORDS];

// Conservative for bilinear lookups on mip 0, where the alpha test of a hit reads.
void bake_opacity(const MaterialSystem material_system,
                  const MicromapBakeJob job,
                  const MicromapBakeBlock block,
                  const float2 uv0,
                  const float2 uv1,
                  const float2 uv2,
                  const uint thread) {
    const MaterialID material_id = job.geometry.material_id;
    const TexelGrid grid = TexelGrid(float2(float(job.alpha_width), float(job.alpha_height)),
                                     float(1 + MERIAN_MICROMAP_TEXEL_SLACK), uv0, uv1, uv2);
    const ExplicitLODSampler lod = ExplicitLODSampler(0.f);
    const MicromapDataAddress states = job.opacity + block.opacity_word_offset;

    // 1. the whole triangle
    if (thread == 0)
        opacity_seen = 0;
    GroupMemoryBarrierWithGroupSync();
    uint dummy;
    InterlockedOr(opacity_seen,
                  scan(material_system, material_id, MicromapKind::Opacity, grid, lod,
                       grid.triangle_footprint(), int(thread), int(BAKE_GROUP_SIZE)),
                  dummy);
    GroupMemoryBarrierWithGroupSync();

    const uint level = job.subdivision_level;
    const uint micro_triangles = 1u << (2 * level);
    const uint words = max(1u, micro_triangles / OMM_STATES_PER_WORD);
    const uint triangle_state = omm_state(opacity_seen);
    if (triangle_state != OMM_STATE_UNKNOWN_OPAQUE) {
        uint packed = 0;
        for (uint i = 0; i < OMM_STATES_PER_WORD; i++) {
            packed |= triangle_state << (i * OMM_STATE_BITS);
        }
        for (uint word = thread; word < words; word += BAKE_GROUP_SIZE) {
            states[word] = packed;
        }
        return;
    }

    // 2. every micro-triangle
    for (uint base = 0; base < micro_triangles; base += BAKE_GROUP_SIZE) {
        if (thread < OMM_WORDS_PER_GROUP)
            omm_words[thread] = 0;
        GroupMemoryBarrierWithGroupSync();

        const uint index = base + thread;
        if (index < micro_triangles) {
            const TexelFootprint footprint = grid.footprint(micromap_index_to_bary(index, level));
            const uint state = omm_state(scan(material_system, material_id, MicromapKind::Opacity,
                                              grid, lod, footprint, 0, 1));
            InterlockedOr(omm_words[thread / OMM_STATES_PER_WORD],
                          state << ((thread % OMM_STATES_PER_WORD) * OMM_STATE_BITS), dummy);
        }
        GroupMemoryBarrierWithGroupSync();

        const uint word = base / OMM_STATES_PER_WORD + thread;
        if (thread < OMM_WORDS_PER_GROUP && word < words)
            states[word] = omm_words[thread];
    }
}

// Classified on a coarse mip: a texel there emits where any texel under it does.
void bake_emission(const MaterialSystem material_system,
                   const MicromapBakeJob job,
                   const MicromapBakeBlock block,
                   const float2 uv0,
                   const float2 uv1,
                   const float2 uv2,
                   const uint thread) {
    const MaterialID material_id = job.geometry.material_id;
    const uint level = job.subdivision_level;
    const float2 finest_size = float2(float(job.emission_width), float(job.emission_height));
    const float2 e1 = (uv1 - uv0) * finest_size;
    const float2 e2 = (uv2 - uv0) * finest_size;
    const float2 e3 = e2 - e1;
    const float extent =
        max(max(abs(e1.x), abs(e1.y)), max(max(abs(e2.x), abs(e2.y)), max(abs(e3.x), abs(e3.y))));
    const uint mip = emission_scan_mip(extent, level, job.emission_max_mip);
    const TexelGrid grid = TexelGrid(
        max(floor(finest_size / float(1u << mip)), float2(1.f)),
        lookup_reach((job.flags & MicromapBakeFlags::EmissionNearestFilter) != 0), uv0, uv1, uv2);
    const ExplicitLODSampler lod = ExplicitLODSampler(float(mip));
    const bool alpha_tested = job.alpha_width != 0;
    const TexelGrid alpha_grid = TexelGrid(
        float2(float(job.alpha_width), float(job.alpha_height)),
        lookup_reach((job.flags & MicromapBakeFlags::AlphaNearestFilter) != 0), uv0, uv1, uv2);
    const ExplicitLODSampler lod0 = ExplicitLODSampler(0.f);

    // 1. the whole triangle
    if (thread == 0)
        emission_seen = 0;
    if (thread < EMISSION_MAX_BIT_WORDS)
        emission_words[thread] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint dummy;
    InterlockedOr(emission_seen,
                  scan(material_system, material_id, MicromapKind::Emission, grid, lod,
                       grid.triangle_footprint(), int(thread), int(BAKE_GROUP_SIZE)),
                  dummy);
    GroupMemoryBarrierWithGroupSync();

    const uint micro_triangles = 1u << (2 * level);
    const uint triangle_emission = settle_emission(emission_seen);
    const bool emission_mixed = triangle_emission == (SEEN_EMITTING | SEEN_DARK);
    const bool may_emit = (triangle_emission & SEEN_EMITTING) != 0;
    if (may_emit && !emission_mixed && !alpha_tested) {
        if (thread < emission_bit_words(level))
            emission_words[thread] = micro_triangles >= 32 ? ~0u : (1u << micro_triangles) - 1u;
    } else if (may_emit) {
        // 2. every micro-triangle
        for (uint index = thread; index < micro_triangles; index += BAKE_GROUP_SIZE) {
            const MicroTriangle bary = micromap_index_to_bary(index, level);
            bool emits = true;
            if (emission_mixed)
                emits = (settle_emission(scan(material_system, material_id, MicromapKind::Emission,
                                              grid, lod, grid.footprint(bary), 0, 1)) &
                         SEEN_EMITTING) != 0;
            if (emits && alpha_tested)
                emits = scan(material_system, material_id, MicromapKind::Cutout, alpha_grid, lod0,
                             alpha_grid.footprint(bary), 0, 1) != SEEN_TRANSPARENT;
            if (emits) {
                const uint grid_index = emission_grid_index_of_micro_triangle(bary, level);
                InterlockedOr(emission_words[grid_index / 32], 1u << (grid_index % 32), dummy);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (thread != 0)
        return;
    const MicromapDataAddress record = job.emission + block.emission_word_offset;
    uint count = 0;
    for (uint word = 0; word < emission_bit_words(level); word++) {
        count += countbits(emission_words[word]);
        record[EMISSION_RECORD_BITS + word] = emission_words[word];
    }
    record[EMISSION_RECORD_COUNT] = count;
}

[shader("compute")]
[numthreads(BAKE_GROUP_SIZE, 1, 1)]
void main(uint3 group_id: SV_GroupID,
          uint3 local_id: SV_GroupThreadID,
          ParameterBlock<MaterialSystem> material_system,
          ParameterBlock<MicromapBakeBinding> params) {
    if (group_id.x >= params.block_count)
        return;
    const uint thread = local_id.x;

    const MicromapBakeBlock block = params.blocks[group_id.x];
    const MicromapBakeJob job = params.jobs[block.job_index];

    float2 uv0;
    float2 uv1;
    float2 uv2;
    get_triangle_uv(job.geometry, block.primitive_id, uv0, uv1, uv2);

    if (job.opacity != nullptr)
        bake_opacity(material_system, job, block, uv0, uv1, uv2, thread);
    if (job.emission != nullptr)
        bake_emission(material_system, job, block, uv0, uv1, uv2, thread);
}
