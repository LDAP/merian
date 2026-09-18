#include "merian-shaders/utils/math-constants.slangh"

__exported import merian_shaders.scene.light_data;
__exported import merian_shaders.scene.environment_map;
__exported import merian_shaders.scene.env_importance;
import merian_shaders.scene.hit;
import merian_shaders.shading.materials.material_system;
import merian_shaders.utils.random;
import merian_shaders.utils.sampling;
import merian_shaders.colors.colorspaces;

namespace merian {

// Renderer-side selection of how direct light is estimated.
static const int NEE_MODE_OFF = 0;
// The light technique replaces the scatter sample: one ray per vertex, mixture density.
static const int NEE_MODE_MIXTURE = 1;
// Light candidates resampled into one shadow ray, MIS-weighted against the scatter sample.
static const int NEE_MODE_RESAMPLED = 2;

// light sampling is wasted below this lobe width: the BSDF is far more peaked than the light
static const float NEE_MIN_ALPHA = 0.05f;

// Scenes whose environment does not emit compile the importance warp out entirely.
extern static const bool merian_scene_has_env_light = false;
extern static const int merian_nee_pool_candidates = 1;
extern static const int merian_nee_cell_candidates = 0;

struct LightSample {
    float3 wo;  // from the receiver towards the light
    float dist; // SCENE_RAY_TMAX for the environment
    float3 Li;  // radiance arriving along -wo
    float pdf;  // solid angle, of the point on the light alone
    // point on the light, offset towards the receiver so a ray aimed at it ends in front of the
    // emitting side; for the environment a point along wo at ENV_MAP_DISTANCE
    float3 pos;
    // emissive triangle, or LIGHT_ENV_INDEX; identifies the sampled light at a hit
    uint index;
}

// A point on a light without its radiance.
struct LightPoint {
    // the sampled point itself (not offset): a ray aimed at it must reach the light, also at
    // grazing angles. For the environment a point along the sampled direction at ENV_MAP_DISTANCE.
    float3 pos;
    uint index;
}

// The light-sampling side of a hit: which light it is (LIGHT_INVALID_INDEX for a non-emitter)
// and the density the light techniques would have generated it with.
struct LightHit {
    uint index;
    float pdf;

    __init() {
        index = LIGHT_INVALID_INDEX;
        pdf = 0.f;
    }
}

// The emissive triangles and the flux distribution over them.
struct LightTable {
    StructuredBuffer<EmissiveTriangle, ScalarDataLayout> triangles;
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

// Lights drawn from the flux distribution once per frame.
struct LightPool {
    StructuredBuffer<uint> entries;
    uint size;
}

// Per-frame world-space lists of the light sources that reach a cell.
struct LightGrid {
    StructuredBuffer<uint> slots;
    StructuredBuffer<LightGridInfo, ScalarDataLayout> cascades;
    // cells the lookup is offset by, so the cell boundaries do not show
    float jitter;
    // how often a draw takes the cell's list; the rest falls back to the pool
    float share;
}

// The environment map as a light.
struct EnvLight {
    EnvImportance importance;
    // texels drawn from importance this frame; empty while the descent is used per sample
    StructuredBuffer<uint> pool;
    uint pool_size;
    // share of the light samples the environment takes, written by the preprocess
    StructuredBuffer<float> split;
}

// Next event estimation over the scene's emissive triangles and the environment map. All
// densities are per solid angle at the receiver.
struct NEE {
    LightTable table;
    LightPool pool;
    LightGrid grid;
    EnvLight env;
    uint selection;
    // UseEnvMap geometry exists: rays towards the environment must find the closest hit
    bool has_sky_portals;

    // One load from the per-frame texel pool where it exists, the pyramid descent otherwise.
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

    bool has_lights() {
        return get_env_share() > 0.f || has_emissive_triangles();
    }

    // How often the environment is chosen over the triangles.
    float get_env_share() {
        if (!merian_scene_has_env_light || !env.importance.is_valid())
            return 0.f;
        const float p_env = env.split[0];
        return has_emissive_triangles() ? p_env : select(p_env > 0.f, 1.f, 0.f);
    }

    // --- lights ---

    // Cell holding pos, or LIGHT_INVALID_INDEX beyond the last cascade. jitter softens the cell
    // boundaries; the caller has to reuse the cell it returns for the density.
    uint get_grid_cell(const float3 pos, const float3 jitter = float3(0)) {
        if (selection != LightSelection::LightSelectionGrid)
            return LIGHT_INVALID_INDEX;
        const LightGridInfo base = grid.cascades[0];
        if (base.dimension == 0)
            return LIGHT_INVALID_INDEX;

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
            const LightGridInfo info = grid.cascades[cascade];
            const float3 local = (pos - info.origin) * info.inv_cell_size + jitter;
            const int3 coord = int3(floor(local));
            if (all(coord >= 0) && all(coord < int(info.dimension))) {
                const uint cell = cascade * cells_per_cascade + uint(coord.x) +
                                  info.dimension * (uint(coord.y) + info.dimension * uint(coord.z));
                return cell;
            }
        }
        return LIGHT_INVALID_INDEX;
    }

    // A vertex always needs a technique to draw from, so the pool covers a split of nothing.
    uint pool_candidates() {
        const uint cell = cell_candidates();
        return uint(max(merian_nee_pool_candidates, cell > 0 ? 0 : 1));
    }

    uint cell_candidates() {
        return selection == LightSelection::LightSelectionGrid
                   ? uint(max(merian_nee_cell_candidates, 0))
                   : 0u;
    }

    uint candidate_count() {
        return pool_candidates() + cell_candidates();
    }

    bool is_grid_cell(const uint bucket) {
        return bucket != LIGHT_INVALID_INDEX;
    }

    // Flux of a geometry, and of the triangles before it, from the prefix sum.
    void geometry_flux(const uint geometry, out float base, out float total) {
        const LightGeometry lg = table.geometries[geometry];
        base = select(lg.first_triangle > 0, table.cdf[max(lg.first_triangle, 1u) - 1u], 0.f);
        total = table.cdf[lg.first_triangle + lg.primitive_count - 1u] - base;
    }

    Optional<uint> get_triangle_index(const GeometryID geometry_id, const uint primitive_id) {
        if (geometry_id >= table.geometry_offset_count)
            return none;
        const uint offset = table.geometry_offsets[geometry_id];
        if (offset == LIGHT_INVALID_INDEX)
            return none;
        return offset + primitive_id;
    }

    // --- selection techniques ---
    //
    // The pool reaches every light in the scene; a cell's list reaches the few that matter where
    // the point stands, and hands a cell it cannot serve back to the pool. Every draw is charged
    // the density of both.

    // One draw from the pool: a light anywhere in the scene, by flux.
    uint draw_pool(inout RandomGenerator rng) {
        if (selection == LightSelection::LightSelectionPower || pool.size == 0)
            return merian::search_flux(table.cdf, rng.next_float() * get_total_flux(), 0,
                                       table.triangle_count - 1);
        return pool.entries[min(uint(rng.next_float() * pool.size), pool.size - 1)];
    }

    // Its density, from the stored prefix sums: with many lights a sum loses the flux in
    // rounding, and the difference of two neighbours is what the search actually realizes.
    float pdf_pool(const uint index) {
        const float total = get_total_flux();
        if (!(total > 0.f))
            return 0.f;
        const float gap = index == 0 ? table.cdf[0] : table.cdf[index] - table.cdf[index - 1];
        return max(gap, 0.f) / total;
    }

    // One draw from the cell's list: a source uniformly among the slots that hold one, then a
    // triangle of it by flux.
    uint draw_grid(const uint cell, inout RandomGenerator rng) {
        uint geometry = LIGHT_INVALID_INDEX;
        if (is_grid_cell(cell) && rng.next_float() < grid.share) {
            const uint base = light_bucket_base(cell);
            uint occupied = 0;
            for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
                const uint held = grid.slots[base + slot];
                // a slot the preprocess never wrote indexes the table out of bounds
                if (held >= table.geometry_count)
                    continue;
                occupied++;
                if (rng.next_float() * float(occupied) < 1.f)
                    geometry = held;
            }
        }
        if (geometry == LIGHT_INVALID_INDEX)
            return draw_pool(rng);

        float flux_base, flux_total;
        geometry_flux(geometry, flux_base, flux_total);
        if (!(flux_total > 0.f))
            return LIGHT_INVALID_INDEX;
        const LightGeometry lg = table.geometries[geometry];
        return merian::search_flux(table.cdf, flux_base + rng.next_float() * flux_total,
                                   lg.first_triangle, lg.first_triangle + lg.primitive_count - 1u);
    }

    // Its density: the slots holding the light over the slots holding anything, times the light's
    // share of its source's flux.
    float pdf_grid(const uint cell, const uint index) {
        if (!is_grid_cell(cell))
            return pdf_pool(index);

        const float pool = (1.f - grid.share) * pdf_pool(index);
        const uint base = light_bucket_base(cell);
        const uint geometry =
            merian::find_light_geometry(table.geometries, table.geometry_count, index);
        uint occupied = 0;
        uint hits = 0;
        for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
            const uint held = grid.slots[base + slot];
            if (held >= table.geometry_count)
                continue;
            occupied++;
            hits += select(held == geometry, 1u, 0u);
        }
        if (occupied == 0)
            return pdf_pool(index);
        if (hits == 0)
            return pool;

        float flux_base, flux_total;
        geometry_flux(geometry, flux_base, flux_total);
        if (!(flux_total > 0.f))
            return 0.f;
        const float triangle =
            table.cdf[index] - select(index > 0, table.cdf[max(index, 1u) - 1u], 0.f);
        return pool +
               grid.share * (float(hits) / float(occupied)) * max(triangle, 0.f) / flux_total;
    }

    // Uniform point on triangle index; Li is the material emission at the point.
    Optional<LightSample> sample_triangle(const uint index,
                                          const float3 pos,
                                          inout RandomGenerator rng,
                                          const MaterialSystem material_system,
                                          const float time) {
        const EmissiveTriangle tri = table.triangles[index];
        const float3 bary = merian::sample_triangle(rng);
        const float3 light_pos = tri.get_position(bary);
        const float2 uv = tri.get_uv(bary);
        // a cutout texel is no surface: the tracer passes through it, so it emits nothing
        if (material_system.alpha_test(tri.material_id, uv))
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

        // the material decides whether a back face emits; culled back faces are never visible
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
        // area measure to solid angle
        ls.pdf = dist2 / (cos_l * 0.5f * area2);
        ls.pos = offset_ray_origin(light_pos, n);
        ls.index = index;
        return ls;
    }

    // Solid angle density of sample_triangle(index) generating the point where the ray from
    // origin along dir meets it. Uses the ray-plane distance, so no hit position is needed.
    float pdf_triangle(const uint index, const float3 origin, const float3 dir) {
        const EmissiveTriangle tri = table.triangles[index];
        const float3 scaled_n = tri.get_scaled_normal();
        const float area2 = length(scaled_n);
        if (area2 <= 0.f)
            return 0.f;
        const float3 n = scaled_n / area2;
        // the front side faces the ray when it approaches against the normal
        const float ndotd = dot(n, dir);
        const float cos_l = tri.is_two_sided() ? abs(ndotd) : -ndotd;
        const float dist = dot(tri.p0 - origin, n) / ndotd;
        if (cos_l <= 0.f || !(dist > 0.f))
            return 0.f;
        return dist * dist / (cos_l * 0.5f * area2);
    }

    // Density of the environment sample producing dir, without the share the environment takes.
    float pdf_env_point(const float3 dir) {
        return merian_scene_has_env_light ? env.importance.pdf(dir) : 0.f;
    }

    // Density of the light techniques together for a sample whose point density is known. The
    // denominator of the generalized balance heuristic (Veach; Bitterli et al. 2020), each
    // technique counted as often as it draws.
    float pdf_lights(const uint cell,
                     const uint index,
                     const float point_pdf,
                     const uint m_pool,
                     const uint m_grid) {
        const float p_e = get_env_share();
        if (index == LIGHT_ENV_INDEX)
            return float(m_pool) * p_e * point_pdf;
        const float from_pool = m_pool > 0 ? float(m_pool) * (1.f - p_e) * pdf_pool(index) : 0.f;
        const float from_grid = m_grid > 0 ? float(m_grid) * pdf_grid(cell, index) : 0.f;
        return (from_pool + from_grid) * point_pdf;
    }

    float pdf_lights(const uint cell, const uint index, const float point_pdf) {
        return pdf_lights(cell, index, point_pdf, pool_candidates(), cell_candidates());
    }

    // The same where the ray from origin along dir meets the light.
    float pdf_lights(const uint cell,
                     const uint index,
                     const float3 origin,
                     const float3 dir,
                     const uint m_pool,
                     const uint m_grid) {
        const float point_pdf =
            index == LIGHT_ENV_INDEX ? pdf_env_point(dir) : pdf_triangle(index, origin, dir);
        return point_pdf > 0.f ? pdf_lights(cell, index, point_pdf, m_pool, m_grid) : 0.f;
    }

    // One draw from a technique: the environment, or a light that technique offers.
    Optional<LightSample> sample_light<Env : IEnvMap>(const uint bucket,
                                                      const float3 pos,
                                                      inout RandomGenerator rng,
                                                      const MaterialSystem material_system,
                                                      const Env env_map,
                                                      const float time,
                                                      const bool from_grid) {
        // the environment is the pool's to offer; a cell holds the lights that reach it
        if (!from_grid && rng.next_float() < get_env_share()) {
            if (let es = sample_env(rng)) {
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
            return none;
        }
        if (!has_emissive_triangles())
            return none;

        const uint index = from_grid ? draw_grid(bucket, rng) : draw_pool(rng);
        if (index == LIGHT_INVALID_INDEX)
            return none;
        return sample_triangle(index, pos, rng, material_system, time);
    }

    // A point on a light without its radiance, drawn from the techniques in proportion to how
    // often each is used.
    Optional<LightPoint> sample_point(const uint cell,
                                      const float3 pos,
                                      inout RandomGenerator rng,
                                      const uint m_pool,
                                      const uint m_grid) {
        const uint draws = m_pool + m_grid;
        if (draws == 0)
            return none;

        LightPoint lp;
        const bool from_grid = rng.next_float() * float(draws) >= float(m_pool);
        if (!from_grid && rng.next_float() < get_env_share()) {
            if (let es = sample_env(rng)) {
                lp.pos = pos + es.dir * ENV_MAP_DISTANCE;
                lp.index = LIGHT_ENV_INDEX;
                return lp;
            }
            return none;
        }
        if (!has_emissive_triangles())
            return none;

        const uint index = from_grid ? draw_grid(cell, rng) : draw_pool(rng);
        if (index == LIGHT_INVALID_INDEX)
            return none;
        const EmissiveTriangle tri = table.triangles[index];
        const float3 scaled_n = tri.get_scaled_normal();
        if (length(scaled_n) <= 0.f)
            return none;
        const float3 light_pos = tri.get_position(merian::sample_triangle(rng));
        // the emitting side must face the receiver unless the triangle emits both ways
        if (dot(scaled_n, pos - light_pos) <= 0.f && !tri.is_two_sided())
            return none;
        lp.pos = light_pos;
        lp.index = index;
        return lp;
    }
}

// Weighted reservoir over the direct-light candidates of one path vertex. Resampling weight
// p_hat / (M p_light + p_dir) (generalized balance heuristic, Bitterli et al. 2020), with p_hat
// the luminance of the unshadowed contribution.
struct DirectLightReservoir {
    float weight_sum;
    float3 value;
    float3 light_pos;
    // LIGHT_INVALID_INDEX while the continuation leads (no shadow ray)
    uint light_index;

    __init() {
        weight_sum = 0.f;
        value = float3(0);
        light_index = LIGHT_INVALID_INDEX;
    }

    [mutating] void add_light(const LightSample ls,
                              const float3 candidate_value,
                              const float pdf_denominator,
                              inout RandomGenerator rng) {
        const float weight = yuv_luminance(candidate_value) / pdf_denominator;
        if (!(weight > 0.f))
            return;
        weight_sum += weight;
        if (rng.next_float() * weight_sum < weight) {
            value = candidate_value;
            light_pos = ls.pos;
            light_index = ls.index;
        }
    }

    bool needs_shadow_ray() {
        return light_index != LIGHT_INVALID_INDEX;
    }

    // Unshadowed estimate; multiply by the winner's visibility.
    float3 resolve() {
        const float target = yuv_luminance(value);
        return target > 0.f ? value * (weight_sum / target) : float3(0);
    }
}

// What the integrator says a candidate is worth, for the resampling to rank it by.
interface IDirectLightTarget {
    float3 unshadowed(const LightSample ls, inout RandomGenerator rng);

    float scatter_pdf(const LightSample ls);
}

// The sample that survived the resampling, already carrying its unbiased weight.
struct DirectLightSample {
    float3 value;
    float3 light_pos;
    uint light_index;

    bool needs_shadow_ray() {
        return light_index != LIGHT_INVALID_INDEX;
    }
}

// Which light selection serves one path vertex; opaque to the integrator.
struct DirectLightVertex {
    uint bucket;
    float3 pos;
}

} // namespace merian
