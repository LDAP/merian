#include "merian-shaders/utils/math-constants.slangh"

__exported import merian_shaders.scene.light_data;
__exported import merian_shaders.scene.environment_map;
__exported import merian_shaders.scene.env_importance;
import merian_shaders.scene.hit;
import merian_shaders.shading.materials.material_system;
import merian_shaders.utils.bit_twiddling;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;
import merian_shaders.utils.sampling;
import merian_shaders.colors.colorspaces;

namespace merian {

static const int NEE_MODE_OFF = 0;
static const int NEE_MODE_MIXTURE = 1;
static const int NEE_MODE_RESAMPLED = 2;

static const float NEE_MIN_ALPHA = 0.05f;

extern static const bool merian_scene_has_env_light = false;
extern static const int merian_nee_scene_draws = 1;
extern static const int merian_nee_cell_draws = 0;
extern static const int merian_nee_slot_weighing = 0;
extern static const bool merian_nee_env_regions = false;

struct LightSample {
    float3 wo;  // from the receiver towards the light
    float dist; // SCENE_RAY_TMAX for the environment
    float3 Li;
    float pdf;  // solid angle, of the point on the light alone
    // offset to the emitting side; for the environment at ENV_MAP_DISTANCE along wo
    float3 pos;
    uint index;
    uint grid_entry;
}

struct LightPoint {
    // not offset, so that a ray aimed at it reaches the light at grazing angles
    float3 pos;
    uint index;
}

struct DirectLightVertex {
    uint cell;
    float3 pos;
    float3 normal;

    bool has_cell() {
        return cell != LIGHT_INVALID_INDEX;
    }
}

struct LightCellWeights {
    float importance;
    uint occupied;
    uint draws;
}

struct LightTable {
    StructuredBuffer<EmissiveTriangle, ScalarDataLayout> triangles;
    StructuredBuffer<EmissiveRegion, ScalarDataLayout> regions;
    // inclusive prefix sum of the flux
    StructuredBuffer<float> cdf;
    // the emissive geometries, ordered by their first triangle
    StructuredBuffer<LightGeometry, ScalarDataLayout> geometries;
    // GeometryID -> first triangle, LIGHT_INVALID_INDEX where the geometry does not emit
    StructuredBuffer<uint> geometry_offsets;
    uint triangle_count;
    uint geometry_count;
    uint geometry_offset_count;
}

struct LightPool {
    StructuredBuffer<uint> entries;
    uint size;
}

struct LightGrid {
    StructuredBuffer<uint> slots;
    StructuredBuffer<LightSource, ScalarDataLayout> sources;
    StructuredBuffer<uint> source_of;
    StructuredBuffer<float> source_cdf;
    StructuredBuffer<LightGridInfo, ScalarDataLayout> cascades;
    bool in_use;
    // in cells
    float jitter;
    uint frame;
    float share;
    float even_share;
    StructuredBuffer<float> visibility;
    StructuredBuffer<LightSource, ScalarDataLayout> slot_sources;
    StructuredBuffer<float> probability;
    RWStructuredBuffer<uint> feedback;
    RWStructuredBuffer<uint> discovered;

    uint lookup(const float3 pos) {
        const uint4 h = pcg4d(uint4(asuint(pos), frame));
        const float3 u = float3(uint_to_zero_one_float_rust(h.x), uint_to_zero_one_float_rust(h.y),
                                uint_to_zero_one_float_rust(h.z));
        const float3 offset = jitter * (u - 0.5f);

        if (!in_use)
            return LIGHT_INVALID_INDEX;
        const LightGridInfo base = cascades[0];

        // the cascade is picked from the unjittered position, so the jitter cannot walk the
        // lookup across cascades and change the cell size under it
        const float cell = 1.f / base.inv_cell_size;
        const float3 to_center = abs(pos - (base.origin + 0.5f * float(base.dimension) * cell));
        const float distance = max(to_center.x, max(to_center.y, to_center.z));
        // snapping moves a cascade by up to one cell, so only dimension / 2 - 1 cells are certain
        const float reach = max(0.5f * float(base.dimension) - 1.f, 1.f) * cell;
        uint cascade = uint(max(ceil(log2(max(distance / reach, 1e-6f))), 0.f));

        const uint cells_per_cascade = base.dimension * base.dimension * base.dimension;
        // the cascade the distance implies always contains pos unless the snapping pushed it out,
        // which the next one up covers
        for (uint step = 0; step < 2 && cascade < base.cascade_count; step++, cascade++) {
            const LightGridInfo info = cascades[cascade];
            const float3 local = (pos - info.origin) * info.inv_cell_size + offset;
            const int3 coord = int3(floor(local));
            if (all(coord >= 0) && all(coord < int(info.dimension)))
                return cascade * cells_per_cascade + uint(coord.x) +
                       info.dimension * (uint(coord.y) + info.dimension * uint(coord.z));
        }
        return LIGHT_INVALID_INDEX;
    }

    bool weighs_at_cell() {
        return merian_nee_slot_weighing == int(LightSlotWeighing::LightSlotWeighingCell);
    }

    // the fill folds it into the cell's probabilities
    float vertex_even_share() {
        return weighs_at_cell() ? 0.f : even_share;
    }

    bool weigh_slot(const uint entry, const DirectLightVertex vertex, out float importance) {
        if (weighs_at_cell()) {
            importance = probability[entry];
            return importance > 0.f;
        }
        const LightSource light = slot_sources[entry];
        const bool occupied = light.flux > 0.f;
        importance =
            occupied ? light.flux * light.geometry_term(vertex.pos, vertex.normal, 0.f) : 0.f;
        return occupied;
    }

    float slot_importance(const uint entry, const DirectLightVertex vertex) {
        float importance;
        weigh_slot(entry, vertex, importance);
        return importance;
    }

    LightCellWeights weigh_cell(const DirectLightVertex vertex) {
        LightCellWeights weights = {0.f, 0u, 0u};
        if (!vertex.has_cell())
            return weights;
        const uint base = light_bucket_base(vertex.cell);
        for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
            float importance;
            if (weigh_slot(base + slot, vertex, importance)) {
                weights.occupied++;
                weights.importance += importance;
            }
        }
        return weights;
    }

    // Every draw is a weighted reservoir over the slots, streamed with one rescaled uniform.
    LightCellWeights weigh_cell(const DirectLightVertex vertex,
                                const uint draws,
                                inout RandomGenerator rng) {
        LightCellWeights weights = {0.f, 0u, 0u};
        if (!vertex.has_cell())
            return weights;

        float u[LIGHT_GRID_MAX_DRAWS];
        uint even = 0;
        for (uint draw = 0; draw < draws; draw++) {
            u[draw] = rng.next_float();
            if (vertex_even_share() > 0.f && rng.next_float() < vertex_even_share())
                even |= 1u << draw;
        }

        const uint base = light_bucket_base(vertex.cell);
        for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
            float importance;
            if (!weigh_slot(base + slot, vertex, importance))
                continue;
            weights.occupied++;
            weights.importance += importance;

            // matches slot_probability while nothing delivers
            const float p_even = 1.f / float(weights.occupied);
            const float p_importance =
                weights.importance > 0.f ? importance / weights.importance : p_even;
            for (uint draw = 0; draw < draws; draw++) {
                const float p = (even & (1u << draw)) != 0 ? p_even : p_importance;
                const bool taken = u[draw] < p;
                u[draw] = min(taken ? u[draw] / p : (u[draw] - p) / (1.f - p),
                              1.f - 0.5f * MERIAN_FLT_EPSILON);
                if (taken) {
                    const uint shift = draw * LIGHT_GRID_SLOT_BITS;
                    weights.draws =
                        (weights.draws & ~((LIGHT_GRID_SLOTS - 1u) << shift)) | (slot << shift);
                }
            }
        }
        return weights;
    }

    float slot_probability(const float importance, const LightCellWeights weights) {
        const float total = weights.importance;
        const float even = total > 0.f ? vertex_even_share() : 1.f;
        return (1.f - even) * (total > 0.f ? importance / total : 0.f) +
               even / float(weights.occupied);
    }

    float slot_share(const DirectLightVertex vertex,
                     const uint entry,
                     const Optional<LightCellWeights> weights) {
        if (weighs_at_cell())
            return probability[entry];
        return slot_probability(slot_importance(entry, vertex),
                                weights.hasValue ? weights.value : weigh_cell(vertex));
    }

    uint find_slot(const uint cell, const uint source, out bool lists_any) {
        const uint base = light_bucket_base(cell);
        uint entry = LIGHT_INVALID_INDEX;
        lists_any = false;
        for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
            const uint held = slots[base + slot];
            lists_any |= held != LIGHT_INVALID_INDEX;
            if (held == source)
                entry = base + slot;
        }
        return entry;
    }

    uint draw_slot(const DirectLightVertex vertex,
                   const LightCellWeights weights,
                   const uint draw,
                   inout RandomGenerator rng) {
        if (!vertex.has_cell() || !(rng.next_float() < share))
            return LIGHT_INVALID_INDEX;
        const uint entry =
            light_bucket_base(vertex.cell) +
            ((weights.draws >> (draw * LIGHT_GRID_SLOT_BITS)) & (LIGHT_GRID_SLOTS - 1u));
        return slots[entry] != LIGHT_INVALID_INDEX ? entry : LIGHT_INVALID_INDEX;
    }

    uint draw_in_source(const uint source, inout RandomGenerator rng) {
        return merian::search_flux(source_cdf, rng.next_float(), source,
                                   source + sources[source].triangle_count - 1u);
    }

    float share_in_source(const uint index, const uint source) {
        return source_cdf[index] - select(index > source, source_cdf[max(index, 1u) - 1u], 0.f);
    }

    void count_sample(const uint entry, const bool reached) {
        InterlockedAdd(feedback[2 * entry], 1u);
        if (reached)
            InterlockedAdd(feedback[2 * entry + 1], 1u);
    }

    void discover(const uint cell, const uint source, const uint discovered_slot) {
        discovered[cell * LIGHT_GRID_DISCOVERED + discovered_slot] = source;
    }
}

struct EnvLight {
    EnvImportance importance;
    // texels drawn from importance this frame; empty to descend the pyramid per sample
    StructuredBuffer<uint> pool;
    uint pool_size;
    StructuredBuffer<float> split;
}

// All densities are per solid angle at the receiver.
struct NEE {
    LightTable table;
    LightPool pool;
    LightGrid grid;
    EnvLight env;
    uint selection;
    // UseEnvMap geometry exists: rays towards the environment must find the closest hit
    bool has_sky_portals;
    LightDebugView debug_view;

    bool has_lights() {
        return get_env_share() > 0.f || has_emissive_triangles();
    }

    // Deterministic per point and frame, so a vertex begun again finds the same densities.
    DirectLightVertex begin_vertex(const float3 pos, const float3 normal) {
        DirectLightVertex vertex;
        vertex.pos = pos;
        vertex.normal = normal;
        vertex.cell = grid.lookup(pos);
        return vertex;
    }

    DirectLightVertex begin_vertex(const float3 pos) {
        return begin_vertex(pos, float3(0));
    }

    // Of the light `index` where the ray from origin along dir meets it, summed over every
    // candidate drawn at the vertex.
    float pdf(const DirectLightVertex vertex,
              const uint index,
              const float3 origin,
              const float3 dir) {
        if (index == LIGHT_INVALID_INDEX)
            return 0.f;
        return pdf_lights(vertex, index, origin, dir, scene_draws(), cell_draws());
    }

    // The same at a traced hit; drawn skips the region test for a point the light technique drew.
    float pdf(const DirectLightVertex vertex,
              const uint index,
              const float3 origin,
              const float3 dir,
              const float2 hit_barycentrics,
              const bool drawn) {
        if (index == LIGHT_INVALID_INDEX)
            return 0.f;
        float point_pdf;
        if (index == LIGHT_ENV_INDEX) {
            point_pdf = pdf_env_point(dir);
        } else {
            if (!drawn && !table.regions[index].contains_mesh(hit_barycentrics))
                return 0.f;
            float dist, cos_l;
            point_pdf = intersect_triangle(index, origin, dir, dist, cos_l)
                            ? pdf_triangle_area(index, dist * dist, cos_l)
                            : 0.f;
        }
        return point_pdf > 0.f
                   ? pdf_lights(vertex, index, dir, point_pdf, scene_draws(), cell_draws())
                   : 0.f;
    }

    // The same for a known point on the light, which need not be where the light is now.
    float pdf(const DirectLightVertex vertex,
              const uint index,
              const float3 dir,
              const float dist2,
              const float3 light_normal) {
        if (index == LIGHT_INVALID_INDEX)
            return 0.f;
        const float point_pdf = index == LIGHT_ENV_INDEX
                                    ? pdf_env_point(dir)
                                    : pdf_triangle(index, vertex.pos, dir, dist2, light_normal);
        return point_pdf > 0.f ? pdf_lights(vertex, index, dir, point_pdf) : 0.f;
    }

    // Of a single draw instead of every candidate.
    float mixture_pdf(const float pdf) {
        return pdf / float(max(draw_count(), 1u));
    }

    float3 get_normal(const uint index) {
        return normalize(table.triangles[index].get_scaled_normal());
    }

    // A single draw without radiance; its density is mixture_pdf.
    Optional<LightPoint> sample_point(const DirectLightVertex vertex, inout RandomGenerator rng) {
        const uint m_scene = scene_draws();
        const uint draws = m_scene + cell_draws();
        if (draws == 0)
            return none;

        LightPoint lp;
        const bool from_cell = rng.next_float() * float(draws) >= float(m_scene);
        if (!from_cell) {
            if (rng.next_float() < get_env_share()) {
                if (let es = sample_env(rng)) {
                    lp.pos = vertex.pos + es.dir * ENV_MAP_DISTANCE;
                    lp.index = LIGHT_ENV_INDEX;
                    return lp;
                }
                return none;
            }
            if (!has_emissive_triangles())
                return none;
        }

        uint entry;
        const uint index =
            from_cell ? draw_cell(vertex, grid.weigh_cell(vertex, 1, rng), 0, rng, entry)
                      : draw_scene(rng);
        if (is_env_region(index)) {
            if (let es = env.importance.sample_in_region(index - LIGHT_ENV_REGION_BASE, rng)) {
                lp.pos = vertex.pos + es.dir * ENV_MAP_DISTANCE;
                lp.index = LIGHT_ENV_INDEX;
                return lp;
            }
            return none;
        }
        if (index == LIGHT_INVALID_INDEX)
            return none;
        const EmissiveTriangle tri = table.triangles[index];
        const float3 scaled_n = tri.get_scaled_normal();
        if (length(scaled_n) <= 0.f)
            return none;
        const float3 bary = merian::sample_triangle(rng);
        const float3 light_pos =
            tri.get_position(table.regions[index].to_triangle(bary, rng.next_float()));
        if (dot(scaled_n, vertex.pos - light_pos) <= 0.f && !tri.is_two_sided())
            return none;
        lp.pos = light_pos;
        lp.index = index;
        return lp;
    }

    Optional<EnvSample> sample_env(inout RandomGenerator rng) {
        if (env.pool_size > 0)
            return env.importance.pool_sample(env.pool, env.pool_size, rng);
        return env.importance.sample_direction(rng);
    }

    float get_total_flux() {
        return table.triangle_count > 0 ? table.cdf[table.triangle_count - 1] : 0.f;
    }

    bool has_emissive_triangles() {
        return get_total_flux() > 0.f;
    }

    float get_env_share() {
        if (!merian_scene_has_env_light || !env.importance.is_valid())
            return 0.f;
        const float p_env = env.split[0];
        return has_emissive_triangles() ? p_env : select(p_env > 0.f, 1.f, 0.f);
    }

    uint scene_draws() {
        const uint cell = cell_draws();
        return uint(max(merian_nee_scene_draws, cell > 0 ? 0 : 1));
    }

    uint cell_draws() {
        return grid.in_use ? uint(clamp(merian_nee_cell_draws, 0, int(LIGHT_GRID_MAX_DRAWS))) : 0u;
    }

    uint draw_count() {
        return scene_draws() + cell_draws();
    }

    // From neighbouring prefix sums: with many lights a sum loses the flux in rounding.
    float triangle_flux(const uint index) {
        return max(table.cdf[index] - select(index > 0, table.cdf[max(index, 1u) - 1u], 0.f), 0.f);
    }

    Optional<uint> get_triangle_index(const GeometryID geometry_id, const uint primitive_id) {
        if (geometry_id >= table.geometry_offset_count)
            return none;
        const uint offset = table.geometry_offsets[geometry_id];
        if (offset == LIGHT_INVALID_INDEX)
            return none;
        return offset + primitive_id;
    }

    uint draw_scene(inout RandomGenerator rng) {
        if (selection == LightSelection::LightSelectionPower || pool.size == 0)
            return merian::search_flux(table.cdf, rng.next_float() * get_total_flux(), 0,
                                       table.triangle_count - 1);
        return pool.entries[min(uint(rng.next_float() * pool.size), pool.size - 1)];
    }

    float pdf_scene(const uint index) {
        const float total = get_total_flux();
        return total > 0.f ? triangle_flux(index) / total : 0.f;
    }

    bool lists_env_regions() {
        return merian_nee_env_regions && merian_scene_has_env_light && env.importance.is_valid();
    }

    uint env_region_source(const float3 dir) {
        return lists_env_regions() ? LIGHT_ENV_REGION_BASE + env.importance.region_of(dir)
                                   : LIGHT_INVALID_INDEX;
    }

    bool reaches_source(const uint sampled, const uint hit) {
        if (hit == sampled)
            return true;
        return selection != LightSelection::LightSelectionPower && hit < table.triangle_count &&
               sampled < table.triangle_count && grid.source_of[hit] == grid.source_of[sampled];
    }

    uint listed_source(const uint index, const float3 dir) {
        if (index == LIGHT_ENV_INDEX)
            return env_region_source(dir);
        return index < table.triangle_count ? grid.source_of[index] : LIGHT_INVALID_INDEX;
    }

    void record_discovery(const DirectLightVertex vertex,
                          const uint hit,
                          const float3 dir,
                          inout RandomGenerator rng) {
        if (!vertex.has_cell())
            return;
        const uint source = listed_source(hit, dir);
        if (source != LIGHT_INVALID_INDEX)
            grid.discover(vertex.cell, source,
                          min(uint(rng.next_float() * LIGHT_GRID_DISCOVERED),
                              LIGHT_GRID_DISCOVERED - 1));
    }

    void record_visibility(const DirectLightVertex vertex,
                           const uint index,
                           const float3 dir,
                           const bool visible) {
        if (!vertex.has_cell())
            return;
        const uint source = listed_source(index, dir);
        if (source == LIGHT_INVALID_INDEX)
            return;
        bool lists_any;
        const uint entry = grid.find_slot(vertex.cell, source, lists_any);
        if (entry != LIGHT_INVALID_INDEX)
            grid.count_sample(entry, visible);
        else if (visible && is_env_region(source))
            grid.discover(vertex.cell, source,
                          xxhash32(uint2(source, grid.frame)) % LIGHT_GRID_DISCOVERED);
    }

    uint draw_cell(const DirectLightVertex vertex,
                   const LightCellWeights weights,
                   const uint draw,
                   inout RandomGenerator rng,
                   out uint entry) {
        entry = grid.draw_slot(vertex, weights, draw, rng);
        if (entry == LIGHT_INVALID_INDEX)
            return has_emissive_triangles() ? draw_scene(rng) : LIGHT_INVALID_INDEX;
        const uint source = grid.slots[entry];
        return is_env_region(source) ? source : grid.draw_in_source(source, rng);
    }

    float pdf_cell(const DirectLightVertex vertex,
                   const Optional<LightCellWeights> weights,
                   const uint index,
                   const float p_scene,
                   const uint entry) {
        if (!vertex.has_cell())
            return p_scene;

        const uint source = grid.source_of[index];
        uint source_entry = entry;
        bool lists_any = entry != LIGHT_INVALID_INDEX;
        if (!lists_any)
            source_entry = grid.find_slot(vertex.cell, source, lists_any);
        if (!lists_any)
            return p_scene;
        const float from_scene = (1.f - grid.share) * p_scene;
        if (source_entry == LIGHT_INVALID_INDEX)
            return from_scene;
        const float p_source = grid.slot_share(vertex, source_entry, weights);
        if (!(p_source > 0.f))
            return from_scene;
        return from_scene + grid.share * p_source * grid.share_in_source(index, source);
    }

    float pdf_cell_env(const DirectLightVertex vertex,
                       const Optional<LightCellWeights> weights,
                       const float3 dir,
                       const float point_pdf,
                       const uint entry) {
        if (!lists_env_regions() || !vertex.has_cell())
            return 0.f;
        const uint region = env.importance.region_of(dir);
        uint source_entry = entry;
        if (source_entry == LIGHT_INVALID_INDEX) {
            bool lists_any;
            source_entry = grid.find_slot(vertex.cell, LIGHT_ENV_REGION_BASE + region, lists_any);
            if (source_entry == LIGHT_INVALID_INDEX)
                return 0.f;
        }
        return grid.share * grid.slot_share(vertex, source_entry, weights) * point_pdf /
               env.importance.region_probability(region);
    }

    Optional<LightSample> sample_triangle(const uint index,
                                          const float3 pos,
                                          inout RandomGenerator rng,
                                          const MaterialSystem material_system,
                                          const float time) {
        const EmissiveTriangle tri = table.triangles[index];
        const EmissiveRegion region = table.regions[index];
        const float3 uniform = merian::sample_triangle(rng);
        const float3 bary = region.to_triangle(uniform, rng.next_float());
        const float3 light_pos = tri.get_position(bary);
        const float2 uv = tri.get_uv(bary);
        // a cutout texel is no surface: the tracer passes through it, so it emits nothing
        if (tri.is_alpha_tested() && material_system.alpha_test(tri.material_id, uv))
            return none;

        const float3 scaled_n = tri.get_scaled_normal();
        const float area2 = length(scaled_n);
        if (area2 <= 0.f)
            return none;
        float3 n = scaled_n / area2;

        const float3 to_light = light_pos - pos;
        const float dist2 = dot(to_light, to_light);
        if (dist2 <= 0.f)
            return none;
        const float dist = sqrt(dist2);
        const float3 wo = to_light / dist;

        float cos_l = dot(n, -wo);
        const bool front_facing = cos_l > 0.f;
        if (!front_facing) {
            if (!tri.is_two_sided())
                return none;
            n = -n;
            cos_l = -cos_l;
        }
        if (cos_l <= 0.f)
            return none;

        const ShadingData sd = ShadingData(-wo, light_pos, uv, time, OrthonormalFrame(n), n,
                                           uint16_t(1), front_facing);

        LightSample ls;
        ls.wo = wo;
        ls.dist = dist;
        ls.Li = material_system.get_emission(tri.material_id, sd, DefaultLODSampler());
        ls.pdf = dist2 / (cos_l * 0.5f * area2 * region.get_fraction());
        ls.pos = offset_ray_origin(
            light_pos, n, triangle_spawn_offset(light_pos, tri.p1 - tri.p0, tri.p2 - tri.p0), n);
        ls.index = index;
        ls.grid_entry = LIGHT_INVALID_INDEX;
        return ls;
    }

    float pdf_triangle_area(const uint index, const float dist2, const float cos_l) {
        return dist2 / (cos_l * 0.5f * length(table.triangles[index].get_scaled_normal()) *
                        table.regions[index].get_fraction());
    }

    bool intersect_triangle(const uint index,
                            const float3 origin,
                            const float3 dir,
                            out float dist,
                            out float cos_l) {
        const EmissiveTriangle tri = table.triangles[index];
        const float3 scaled_n = tri.get_scaled_normal();
        const float area2 = length(scaled_n);
        dist = 0.f;
        cos_l = 0.f;
        if (area2 <= 0.f)
            return false;
        const float3 n = scaled_n / area2;
        const float ndotd = dot(n, dir);
        cos_l = tri.is_two_sided() ? abs(ndotd) : -ndotd;
        dist = dot(tri.p0 - origin, n) / ndotd;
        return cos_l > 0.f && dist > 0.f;
    }

    bool region_contains_point(const uint index, const float3 pos) {
        const float2 b = table.triangles[index].get_barycentrics(pos);
        return table.regions[index].contains(float3(1.f - b.x - b.y, b));
    }

    float pdf_triangle(const uint index, const float3 origin, const float3 dir) {
        float dist, cos_l;
        if (!intersect_triangle(index, origin, dir, dist, cos_l) ||
            !region_contains_point(index, origin + dist * dir))
            return 0.f;
        return pdf_triangle_area(index, dist * dist, cos_l);
    }

    float pdf_triangle(const uint index,
                       const float3 origin,
                       const float3 dir,
                       const float dist2,
                       const float3 n) {
        const EmissiveTriangle tri = table.triangles[index];
        const float ndotd = dot(n, dir);
        const float cos_l = tri.is_two_sided() ? abs(ndotd) : -ndotd;
        if (!(cos_l > 0.f) || !region_contains_point(index, origin + sqrt(dist2) * dir))
            return 0.f;
        return pdf_triangle_area(index, dist2, cos_l);
    }

    // Without the environment's share.
    float pdf_env_point(const float3 dir) {
        return merian_scene_has_env_light ? env.importance.pdf(dir) : 0.f;
    }

    // Every technique counted as often as it draws: the denominator of the generalized balance
    // heuristic (Veach 1997; Bitterli et al. 2020).
    float pdf_lights(const DirectLightVertex vertex,
                     const uint index,
                     const float3 dir,
                     const float point_pdf,
                     const uint m_scene,
                     const uint m_cell,
                     const Optional<LightCellWeights> weights = none,
                     const uint entry = LIGHT_INVALID_INDEX) {
        const float p_e = get_env_share();
        if (index == LIGHT_ENV_INDEX) {
            const float from_cell =
                m_cell > 0 ? float(m_cell) * pdf_cell_env(vertex, weights, dir, point_pdf, entry)
                           : 0.f;
            return float(m_scene) * p_e * point_pdf + from_cell;
        }
        const float p_scene = pdf_scene(index);
        const float from_cell =
            m_cell > 0 ? float(m_cell) * pdf_cell(vertex, weights, index, p_scene, entry) : 0.f;
        return (float(m_scene) * (1.f - p_e) * p_scene + from_cell) * point_pdf;
    }

    float pdf_lights(const DirectLightVertex vertex,
                     const uint index,
                     const float3 dir,
                     const float point_pdf) {
        return pdf_lights(vertex, index, dir, point_pdf, scene_draws(), cell_draws());
    }

    float pdf_lights(const DirectLightVertex vertex,
                     const uint index,
                     const float3 origin,
                     const float3 dir,
                     const uint m_scene,
                     const uint m_cell) {
        const float point_pdf =
            index == LIGHT_ENV_INDEX ? pdf_env_point(dir) : pdf_triangle(index, origin, dir);
        return point_pdf > 0.f ? pdf_lights(vertex, index, dir, point_pdf, m_scene, m_cell) : 0.f;
    }

    LightSample env_light_sample<Env : IEnvMap>(const EnvSample es,
                                                const float3 pos,
                                                const MaterialSystem material_system,
                                                const Env env_map,
                                                const float time,
                                                const uint entry) {
        LightSample ls;
        ls.wo = es.dir;
        ls.dist = SCENE_RAY_TMAX;
        ls.Li = env_map.eval(make_env_shading_data(es.dir, pos, time),
                             material_system.texture_manager, DefaultLODSampler());
        ls.pdf = es.pdf;
        ls.pos = pos + es.dir * ENV_MAP_DISTANCE;
        ls.index = LIGHT_ENV_INDEX;
        ls.grid_entry = entry;
        return ls;
    }

    Optional<LightSample> sample_light<Env : IEnvMap>(const DirectLightVertex vertex,
                                                      const LightCellWeights weights,
                                                      inout RandomGenerator rng,
                                                      const MaterialSystem material_system,
                                                      const Env env_map,
                                                      const float time,
                                                      const bool from_cell,
                                                      const uint cell_draw) {
        if (!from_cell) {
            if (rng.next_float() < get_env_share()) {
                if (let es = sample_env(rng))
                    return env_light_sample(es, vertex.pos, material_system, env_map, time,
                                            LIGHT_INVALID_INDEX);
                return none;
            }
            if (!has_emissive_triangles())
                return none;
        }
        uint entry = LIGHT_INVALID_INDEX;
        const uint index =
            from_cell ? draw_cell(vertex, weights, cell_draw, rng, entry) : draw_scene(rng);
        if (is_env_region(index)) {
            if (let es = env.importance.sample_in_region(index - LIGHT_ENV_REGION_BASE, rng))
                return env_light_sample(es, vertex.pos, material_system, env_map, time, entry);
            return none;
        }
        if (index == LIGHT_INVALID_INDEX)
            return none;
        if (let ls = sample_triangle(index, vertex.pos, rng, material_system, time)) {
            LightSample s = ls;
            s.grid_entry = entry;
            return s;
        }
        return none;
    }
}

// Resampling weight p_hat / (M p_light + p_scatter), p_hat the luminance of the unshadowed
// contribution (Bitterli et al. 2020).
struct DirectLightReservoir {
    float weight_sum;
    float3 value;
    // the winner's p_light / (p_light + p_scatter)
    float light_share;
    float3 light_pos;
    uint light_index;

    __init() {
        weight_sum = 0.f;
        value = float3(0);
        light_share = 0.f;
        light_index = LIGHT_INVALID_INDEX;
    }

    [mutating] void add_light(const LightSample ls,
                              const float3 candidate_value,
                              const float light_pdf,
                              const float scatter_pdf,
                              inout RandomGenerator rng) {
        const float weight = yuv_luminance(candidate_value) / (light_pdf + scatter_pdf);
        if (!(weight > 0.f))
            return;
        weight_sum += weight;
        if (rng.next_float() * weight_sum < weight) {
            value = candidate_value;
            light_share = light_pdf / (light_pdf + scatter_pdf);
            light_pos = ls.pos;
            light_index = ls.index;
        }
    }

    bool needs_shadow_ray() {
        return light_index != LIGHT_INVALID_INDEX;
    }

    float3 resolve() {
        const float target = yuv_luminance(value);
        return target > 0.f ? value * (weight_sum / target) : float3(0);
    }

    // 1/W for the target value * light_share (Lin et al. 2022)
    float realized_pdf() {
        return weight_sum > 0.f ? yuv_luminance(value) * light_share / weight_sum : 0.f;
    }
}

interface IDirectLightTarget {
    float3 unshadowed(const LightSample ls, inout RandomGenerator rng);

    float scatter_pdf(const LightSample ls);
}

struct DirectLightSample {
    // unshadowed, already divided by the resampling density
    float3 value;
    float3 light_pos;
    uint light_index;
    float realized_pdf;
}

} // namespace merian
