#include "merian-shaders/utils/math-constants.slangh"

__exported import merian_shaders.scene.light_data;
__exported import merian_shaders.scene.environment_map;
__exported import merian_shaders.scene.env_importance;
import merian_shaders.scene.hit;
import merian_shaders.shading.materials.material_system;
import merian_shaders.utils.bit_twiddling;
import merian_shaders.utils.morton;
import merian_shaders.utils.random;
import merian_shaders.utils.sampling;
import merian_shaders.colors.colorspaces;

namespace merian {

static const float NEE_MIN_ALPHA = 0.05f;

extern static const bool merian_scene_has_env_light = false;
extern static const int merian_nee_draws = 1;
extern static const int merian_nee_source = 0;
extern static const int merian_nee_slot_weighing = 0;

struct LightSample {
    float3 wo;  // from the receiver towards the light
    float dist; // SCENE_RAY_TMAX for the environment
    float3 Li;
    float pdf;  // solid angle, of the point on the light alone
    // offset to the emitting side; for the environment at ENV_MAP_DISTANCE along wo
    float3 pos;
    uint index;
}

struct LightPoint {
    // not offset, so that a ray aimed at it reaches the light at grazing angles
    float3 pos;
    uint index;
    // LIGHT_INVALID_INDEX without cuts
    uint entry;
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
    // the emissive geometries, ordered by their first triangle
    StructuredBuffer<LightGeometry, ScalarDataLayout> geometries;
    // GeometryID -> first triangle, LIGHT_INVALID_INDEX where the geometry does not emit
    StructuredBuffer<uint> geometry_offsets;
    uint triangle_count;
    uint geometry_count;
    uint geometry_offset_count;
}

struct LightTree {
    StructuredBuffer<float2> cdf;
    StructuredBuffer<uint> order;
    StructuredBuffer<uint> rank;
    // 0 where the cells do not list the environment
    uint env_texels;
}

struct LightGrid {
    StructuredBuffer<uint> starts;
    StructuredBuffer<LightGridInfo, ScalarDataLayout> cascades;
    bool in_use;
    // in cells
    float jitter;
    float even_share;
    StructuredBuffer<float3, ScalarDataLayout> contribution;
    StructuredBuffer<float> estimate;
    StructuredBuffer<LightBound, ScalarDataLayout> slot_bounds;
    StructuredBuffer<float> probability;
    RWStructuredBuffer<uint> feedback;
    // per cell: whether any of its slots got feedback
    RWStructuredBuffer<uint> touched;
    uint light_count;

    int3 cell_coord(const uint cascade, const float3 pos, const float3 offset) {
        const LightGridInfo info = cascades[cascade];
        return int3(floor((pos - info.origin) * info.inv_cell_size + offset));
    }

    uint lookup(const float3 pos, inout RandomGenerator rng) {
        if (!in_use)
            return LIGHT_INVALID_INDEX;
        if (merian_nee_source == int(LightSource::LightSourceScene))
            return 0u;
        const LightGridInfo base = cascades[0];

        // the cascade is picked from the unjittered position, so the jitter cannot walk the
        // lookup across cascades and change the cell size under it
        const float cell = 1.f / base.inv_cell_size;
        const float3 to_center = abs(pos - (base.origin + 0.5f * float(base.dimension) * cell));
        const float distance = max(to_center.x, max(to_center.y, to_center.z));
        // snapping moves a cascade by up to one cell, so only dimension / 2 - 1 cells are certain
        const float reach = max(0.5f * float(base.dimension) - 1.f, 1.f) * cell;
        const uint outermost = base.cascade_count - 1u;
        uint cascade = min(uint(max(ceil(log2(max(distance / reach, 1e-6f))), 0.f)), outermost);
        const float3 offset = jitter * (rng.next_float3() - 0.5f);

        // the cascade the distance implies always contains pos unless the snapping pushed it out,
        // which the next one up covers
        int3 coord = cell_coord(cascade, pos, offset);
        if ((any(coord < 0) || any(coord >= int(base.dimension))) && cascade < outermost)
            coord = cell_coord(++cascade, pos, offset);
        return light_cell_index(cascade, uint3(clamp(coord, 0, int(base.dimension) - 1)),
                                base.dimension);
    }

    bool weighs_at_cell() {
        return merian_nee_slot_weighing == int(LightSlotWeighing::LightSlotWeighingCell);
    }

    // the fill folds it into the cell's probabilities
    float vertex_even_share() {
        return weighs_at_cell() ? 0.f : even_share;
    }

    float cell_probability(const uint entry) {
        const bool first = (entry & (LIGHT_GRID_SLOTS - 1u)) == 0;
        return probability[entry] - (first ? 0.f : probability[max(entry, 1u) - 1u]);
    }

    bool weigh_slot(const uint entry, const DirectLightVertex vertex, out float importance) {
        if (weighs_at_cell()) {
            importance = cell_probability(entry);
            return importance > 0.f;
        }
        const LightBound light = slot_bounds[entry];
        const bool occupied = light.flux > 0.f;
        importance = occupied ? light.flux * light.geometry_term(vertex.pos, vertex.normal) : 0.f;
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

        if (weighs_at_cell()) {
            const uint base = light_bucket_base(vertex.cell);
            const float total = probability[base + LIGHT_GRID_SLOTS - 1u];
            if (!(total > 0.f))
                return weights;
            for (uint draw = 0; draw < draws; draw++) {
                const float u = rng.next_float() * total;
                uint lo = 0;
                for (uint step = LIGHT_GRID_SLOTS >> 1; step > 0; step >>= 1) {
                    if (probability[base + lo + step - 1u] <= u)
                        lo += step;
                }
                weights.draws |= lo << (draw * LIGHT_GRID_SLOT_BITS);
            }
            weights.occupied = 1;
            return weights;
        }

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
            return cell_probability(entry);
        return slot_probability(slot_importance(entry, vertex),
                                weights.hasValue ? weights.value : weigh_cell(vertex));
    }

    uint slot_of(const uint cell, const uint position) {
        const uint base = light_bucket_base(cell);
        uint slot = 0;
        for (uint step = LIGHT_GRID_SLOTS >> 1; step > 0; step >>= 1) {
            if (starts[base + slot + step] <= position)
                slot += step;
        }
        return base + slot;
    }

    uint2 slot_range(const uint entry) {
        const bool last = (entry & (LIGHT_GRID_SLOTS - 1u)) == LIGHT_GRID_SLOTS - 1u;
        const uint next = starts[last ? entry : entry + 1u];
        return uint2(starts[entry], last ? light_count : next);
    }

    uint
    draw_slot(const DirectLightVertex vertex, const LightCellWeights weights, const uint draw) {
        if (weights.occupied == 0)
            return LIGHT_INVALID_INDEX;
        return light_bucket_base(vertex.cell) +
               ((weights.draws >> (draw * LIGHT_GRID_SLOT_BITS)) & (LIGHT_GRID_SLOTS - 1u));
    }

    bool draws_from(const DirectLightVertex vertex, const Optional<LightCellWeights> weights) {
        if (!vertex.has_cell())
            return false;
        // an empty cut weighs every slot 0
        if (weighs_at_cell())
            return true;
        return (weights.hasValue ? weights.value : weigh_cell(vertex)).occupied > 0;
    }

    // fixed point, rounded stochastically
    void count_sample(const uint entry, const float contribution, const float2 u) {
        const uint at = LIGHT_GRID_FEEDBACK_STRIDE * entry;
        touched[entry / LIGHT_GRID_SLOTS] = 1u;
        uint counted;
        InterlockedAdd(feedback[at], 1u, counted);
        if (contribution > 0.f && counted < LIGHT_GRID_FEEDBACK_SAMPLES) {
            const float c = min(contribution, LIGHT_GRID_MAX_CONTRIBUTION);
            InterlockedAdd(feedback[at + 1], uint(c * LIGHT_GRID_CONTRIBUTION_SCALE + u.x));
            InterlockedAdd(feedback[at + 2],
                           uint(c * c * LIGHT_GRID_CONTRIBUTION_SQUARE_SCALE + u.y));
        }
    }
}

struct EnvLight {
    EnvImportance importance;
    // texels drawn from importance this frame; empty to search per sample
    StructuredBuffer<uint> pool;
    uint pool_size;
}

// All densities are per solid angle at the receiver.
struct NEE {
    LightTable table;
    LightTree tree;
    LightGrid grid;
    EnvLight env;
    // UseEnvMap geometry exists: rays towards the environment must find the closest hit
    bool has_sky_portals;
    LightDebugView debug_view;

    bool has_lights() {
        return has_cuts() ? has_emissive_triangles() || lists_env() : has_env();
    }

    // Deterministic per point and random numbers, so a vertex begun again finds the same densities.
    DirectLightVertex begin_vertex(const float3 pos,
                                   const float3 normal,
                                   inout RandomGenerator rng) {
        DirectLightVertex vertex;
        vertex.pos = pos;
        vertex.normal = normal;
        vertex.cell = grid.lookup(pos, rng);
        return vertex;
    }

    DirectLightVertex begin_vertex(const float3 pos, inout RandomGenerator rng) {
        return begin_vertex(pos, float3(0), rng);
    }

    // Of the light `index` where the ray from origin along dir meets it, summed over every
    // candidate drawn at the vertex.
    float pdf(const DirectLightVertex vertex,
              const uint index,
              const float3 origin,
              const float3 dir) {
        if (index == LIGHT_INVALID_INDEX)
            return 0.f;
        if (index == LIGHT_ENV_INDEX)
            return pdf_env(vertex, dir);
        const float point_pdf = pdf_triangle(index, origin, dir);
        return point_pdf > 0.f ? pdf_lights(vertex, index, dir, point_pdf) : 0.f;
    }

    // The same at a traced hit; drawn skips the region test for a point the light technique drew.
    float pdf(const DirectLightVertex vertex,
              const uint index,
              const float3 origin,
              const float3 dir,
              const float2 hit_barycentrics,
              const bool drawn,
              const uint entry = LIGHT_INVALID_INDEX) {
        if (index == LIGHT_INVALID_INDEX)
            return 0.f;
        if (index == LIGHT_ENV_INDEX)
            return pdf_env(vertex, dir);
        if (!drawn && !table.regions[index].contains_mesh(hit_barycentrics))
            return 0.f;
        float dist, cos_l;
        const float point_pdf = intersect_triangle(index, origin, dir, dist, cos_l)
                                    ? pdf_triangle_area(index, dist * dist, cos_l)
                                    : 0.f;
        return point_pdf > 0.f ? pdf_lights(vertex, index, dir, point_pdf, none, entry) : 0.f;
    }

    // The same for a known point on the light, which need not be where the light is now.
    float pdf(const DirectLightVertex vertex,
              const uint index,
              const float3 dir,
              const float dist2,
              const float3 light_normal) {
        if (index == LIGHT_INVALID_INDEX)
            return 0.f;
        if (index == LIGHT_ENV_INDEX)
            return pdf_env(vertex, dir);
        const float point_pdf = pdf_triangle(index, vertex.pos, dir, dist2, light_normal);
        return point_pdf > 0.f ? pdf_lights(vertex, index, dir, point_pdf) : 0.f;
    }

    // Of a single draw instead of every candidate.
    float mixture_pdf(const float pdf) {
        return pdf / float(draw_count());
    }

    float3 get_normal(const uint index) {
        return normalize(table.triangles[index].get_scaled_normal());
    }

    // A single draw without radiance; its density is mixture_pdf.
    Optional<LightPoint> sample_point(const DirectLightVertex vertex, inout RandomGenerator rng) {
        uint entry;
        return sample_point(vertex, rng, entry);
    }

    // entry: the cell slot drawn from, also where the draw yields no point.
    Optional<LightPoint>
    sample_point(const DirectLightVertex vertex, inout RandomGenerator rng, out uint entry) {
        entry = LIGHT_INVALID_INDEX;
        uint texel = 0;
        const uint index =
            has_cuts() ? draw_cell(vertex, grid.weigh_cell(vertex, 1, rng), 0, rng, entry, texel)
                       : LIGHT_ENV_INDEX;
        return light_point(vertex, index, texel, entry, rng);
    }

    Optional<LightPoint> light_point(const DirectLightVertex vertex,
                                     const uint index,
                                     const uint texel,
                                     const uint entry,
                                     inout RandomGenerator rng) {
        LightPoint lp;
        lp.entry = entry;
        if (index == LIGHT_ENV_INDEX) {
            if (let es = draw_env(texel, rng)) {
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

    Optional<EnvSample> draw_env(const uint texel, inout RandomGenerator rng) {
        if (has_cuts())
            return env.importance.texel_sample(morton_decode2d(texel), rng.next_float2());
        if (!has_env())
            return none;
        if (env.pool_size > 0)
            return env.importance.pool_sample(env.pool, env.pool_size, rng);
        return env.importance.sample_direction(rng);
    }

    float get_total_flux() {
        if (table.triangle_count == 0)
            return 0.f;
        const float2 total = tree.cdf[table.triangle_count - 1];
        return total.x + total.y;
    }

    bool has_emissive_triangles() {
        return get_total_flux() > 0.f;
    }

    bool has_env() {
        return merian_scene_has_env_light && env.importance.is_valid();
    }

    bool has_cuts() {
        return grid.in_use;
    }

    uint draw_count() {
        return uint(merian_nee_draws);
    }

    Optional<uint> get_triangle_index(const GeometryID geometry_id, const uint primitive_id) {
        if (geometry_id >= table.geometry_offset_count)
            return none;
        const uint offset = table.geometry_offsets[geometry_id];
        if (offset == LIGHT_INVALID_INDEX)
            return none;
        return offset + primitive_id;
    }

    bool lists_env() {
        return has_env() && tree.env_texels > 0;
    }

    uint tree_position(const uint index, const float3 dir) {
        if (index == LIGHT_ENV_INDEX)
            return lists_env()
                       ? table.triangle_count + morton_encode2d(env.importance.texel_of(dir))
                       : LIGHT_INVALID_INDEX;
        return index < table.triangle_count ? tree.rank[index] : LIGHT_INVALID_INDEX;
    }

    float nominal_radiance(const uint index, const float3 dir) {
        if (index == LIGHT_ENV_INDEX)
            return env.importance.value(dir);
        const EmissiveTriangle tri = table.triangles[index];
        const float area = 0.5f * length(tri.get_scaled_normal()) *
                           table.regions[index].get_fraction() * (tri.is_two_sided() ? 2.f : 1.f);
        return area > 0.f ? tri.flux / (MERIAN_PI * area) : 0.f;
    }

    float delivery(const uint index, const float3 dir, const float radiance) {
        if (!(radiance > 0.f))
            return 0.f;
        const float nominal = nominal_radiance(index, dir);
        return nominal > 0.f ? radiance / nominal : 1.f;
    }

    bool reaches(const uint sampled, const uint entry, const uint hit) {
        if (hit == sampled)
            return true;
        if (entry == LIGHT_INVALID_INDEX || hit >= table.triangle_count)
            return false;
        const uint2 range = grid.slot_range(entry);
        const uint position = tree.rank[hit];
        return position >= range.x && position < range.y;
    }

    void record_failed_draw(const uint entry) {
        if (entry != LIGHT_INVALID_INDEX)
            grid.count_sample(entry, 0.f, float2(0));
    }

    void record_contribution(const DirectLightVertex vertex,
                             const uint index,
                             const float3 dir,
                             const float radiance,
                             const float dist,
                             inout RandomGenerator rng,
                             uint entry = LIGHT_INVALID_INDEX) {
        if (!vertex.has_cell() || index == LIGHT_INVALID_INDEX)
            return;
        const uint position = tree_position(index, dir);
        if (position == LIGHT_INVALID_INDEX)
            return;
        if (entry == LIGHT_INVALID_INDEX)
            entry = grid.slot_of(vertex.cell, position);
        float contribution = delivery(index, dir, radiance);
        if (contribution > 0.f && index != LIGHT_ENV_INDEX)
            contribution *= grid.estimate[entry] / max(dist * dist, 1e-8f);
        grid.count_sample(entry, contribution, rng.next_float2());
    }

    uint draw_cell(const DirectLightVertex vertex,
                   const LightCellWeights weights,
                   const uint draw,
                   inout RandomGenerator rng,
                   out uint entry,
                   out uint texel) {
        texel = 0;
        entry = grid.draw_slot(vertex, weights, draw);
        return entry != LIGHT_INVALID_INDEX ? draw_in_slot(entry, rng, texel) : LIGHT_INVALID_INDEX;
    }

    uint draw_in_slot(const uint entry, inout RandomGenerator rng, out uint texel) {
        texel = 0;
        const uint2 range = grid.slot_range(entry);
        const uint count = table.triangle_count;
        if (range.x >= range.y)
            return LIGHT_INVALID_INDEX;
        if (range.x >= count) {
            if (!lists_env())
                return LIGHT_INVALID_INDEX;
            texel = search_range(env.importance.cdf, range.x - count, range.y - count,
                                 rng.next_float2());
            return LIGHT_ENV_INDEX;
        }
        return tree.order[search_range(tree.cdf, range.x, range.y, rng.next_float2())];
    }

    float pdf_cell(const DirectLightVertex vertex,
                   const Optional<LightCellWeights> weights,
                   const uint index,
                   uint entry) {
        if (!grid.draws_from(vertex, weights))
            return 0.f;
        const uint position = tree.rank[index];
        if (entry == LIGHT_INVALID_INDEX)
            entry = grid.slot_of(vertex.cell, position);
        const uint2 range = grid.slot_range(entry);
        return grid.slot_share(vertex, entry, weights) *
               range_share(tree.cdf, range.x, range.y, position);
    }

    float pdf_cell_env(const DirectLightVertex vertex,
                       const Optional<LightCellWeights> weights,
                       const float3 dir) {
        if (!lists_env() || !grid.draws_from(vertex, weights))
            return 0.f;
        const uint count = table.triangle_count;
        const uint texel = morton_encode2d(env.importance.texel_of(dir));
        const uint entry = grid.slot_of(vertex.cell, count + texel);
        const uint2 range = grid.slot_range(entry);
        if (range.x < count)
            return 0.f;
        return grid.slot_share(vertex, entry, weights) *
               range_share(env.importance.cdf, range.x - count, range.y - count, texel) *
               float(tree.env_texels) / MERIAN_FOUR_PI;
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

    float pdf_env_point(const float3 dir) {
        return merian_scene_has_env_light ? env.importance.pdf(dir) : 0.f;
    }

    // Summed over the draws: the denominator of the generalized balance heuristic (Veach 1997;
    // Bitterli et al. 2020).
    float pdf_lights(const DirectLightVertex vertex,
                     const uint index,
                     const float3 dir,
                     const float point_pdf,
                     const Optional<LightCellWeights> weights = none,
                     const uint entry = LIGHT_INVALID_INDEX) {
        if (index == LIGHT_ENV_INDEX)
            return pdf_env(vertex, dir, weights);
        if (!has_cuts())
            return 0.f;
        return float(draw_count()) * pdf_cell(vertex, cell_weights(vertex, weights), index, entry) *
               point_pdf;
    }

    float pdf_env(const DirectLightVertex vertex,
                  const float3 dir,
                  const Optional<LightCellWeights> weights = none) {
        return float(draw_count()) * (has_cuts()
                                          ? pdf_cell_env(vertex, cell_weights(vertex, weights), dir)
                                          : pdf_env_point(dir));
    }

    Optional<LightCellWeights> cell_weights(const DirectLightVertex vertex,
                                            const Optional<LightCellWeights> weights) {
        if (weights.hasValue || grid.weighs_at_cell())
            return weights;
        return grid.weigh_cell(vertex);
    }

    LightSample env_light_sample<Env : IEnvMap>(const EnvSample es,
                                                const float3 pos,
                                                const MaterialSystem material_system,
                                                const Env env_map,
                                                const float time) {
        LightSample ls;
        ls.wo = es.dir;
        ls.dist = SCENE_RAY_TMAX;
        ls.Li = env_map.eval(make_env_shading_data(es.dir, pos, time),
                             material_system.texture_manager, DefaultLODSampler());
        ls.pdf = es.pdf;
        ls.pos = pos + es.dir * ENV_MAP_DISTANCE;
        ls.index = LIGHT_ENV_INDEX;
        return ls;
    }

    // entry: the cell slot drawn from, also where the draw yields no sample.
    Optional<LightSample> sample_light<Env : IEnvMap>(const DirectLightVertex vertex,
                                                      const LightCellWeights weights,
                                                      inout RandomGenerator rng,
                                                      const MaterialSystem material_system,
                                                      const Env env_map,
                                                      const float time,
                                                      const uint draw,
                                                      out uint entry) {
        entry = LIGHT_INVALID_INDEX;
        uint texel = 0;
        const uint index =
            has_cuts() ? draw_cell(vertex, weights, draw, rng, entry, texel) : LIGHT_ENV_INDEX;
        if (index == LIGHT_ENV_INDEX) {
            if (let es = draw_env(texel, rng))
                return env_light_sample(es, vertex.pos, material_system, env_map, time);
            return none;
        }
        if (index == LIGHT_INVALID_INDEX)
            return none;
        return sample_triangle(index, vertex.pos, rng, material_system, time);
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
    float light_radiance;

    __init() {
        weight_sum = 0.f;
        value = float3(0);
        light_share = 0.f;
        light_index = LIGHT_INVALID_INDEX;
        light_radiance = 0.f;
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
            light_radiance = yuv_luminance(ls.Li);
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
    float radiance;
}

} // namespace merian
