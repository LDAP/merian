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

// Scenes whose environment does not emit compile the importance warp out entirely. Set from
// EnvMap::is_emissive() (see scene.cpp).
extern static const bool merian_scene_has_env_light = false;

struct LightSample {
    float3 wo;  // from the receiver towards the light
    float dist; // SCENE_RAY_TMAX for the environment
    float3 Li;  // radiance arriving along -wo
    float pdf;  // solid angle, includes the light selection
    // point on the light, offset towards the receiver so a ray aimed at it ends in front of the
    // emitting side; for the environment a point along wo at ENV_MAP_DISTANCE
    float3 pos;
    // emissive triangle, or LIGHT_ENV_INDEX; identifies the sampled light at a hit
    uint index;

    bool is_distant() {
        return index == LIGHT_ENV_INDEX;
    }
}

// A point on a light without its radiance: all the mixture estimator needs, since it reads the
// emission off the vertex its scatter ray reaches.
struct LightPoint {
    // the sampled point itself (not offset): a ray aimed at it must reach the light, also at
    // grazing angles. For the environment a point along the sampled direction at ENV_MAP_DISTANCE.
    float3 pos;
    uint index;

    bool is_distant() {
        return index == LIGHT_ENV_INDEX;
    }
}

// The light-sampling side of a hit: which light it is (LIGHT_INVALID_INDEX for a non-emitter)
// and the density with which NEE::sample would have generated it.
struct LightHit {
    uint index;
    float pdf;

    __init() {
        index = LIGHT_INVALID_INDEX;
        pdf = 0.f;
    }
}

// Next event estimation over the scene's emissive triangles and the environment map. Triangles
// are selected proportional to their flux, points uniformly by area; the environment is chosen
// with probability p_env. All densities are per solid angle at the receiver.
struct NEE {
    StructuredBuffer<EmissiveTriangle, ScalarDataLayout> triangles;
    // inclusive prefix sum of the selection weights
    StructuredBuffer<float> cdf;
    // per-triangle selection weight; the last cdf entry normalizes it
    StructuredBuffer<float> weights;
    // per-frame draws from the weights, in tiles: every pixel of a screen tile draws from the
    // same one, so the entries it touches stay in cache
    StructuredBuffer<uint> pool;
    uint pool_tile_size;
    // LIGHT_GRID_SLOTS lights per cell, resampled towards the cell every frame
    StructuredBuffer<uint> grid;
    // GeometryID -> first triangle index, LIGHT_INVALID_INDEX for non-emissive geometry
    StructuredBuffer<uint> geometry_light_offsets;
    StructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info;
    // probability of taking the cell list where one exists; the rest falls back to the pool so
    // that no light is ever unreachable
    float grid_probability;
    uint pool_size;
    uint selection;
    uint triangle_count;
    uint geometry_count;
    // 0 if the environment is not sampled
    float p_env;
    // UseEnvMap geometry exists: rays towards the environment must find the closest hit
    bool has_sky_portals;
    EnvImportance env_importance;
    // texels drawn from env_importance this frame; empty while the descent is used per sample
    StructuredBuffer<uint> env_pool;
    uint env_pool_size;

    // One load from the per-frame texel pool where it exists, the pyramid descent otherwise. Both
    // draw the same distribution.
    Optional<EnvSample> sample_env(inout RandomGenerator rng) {
        if (env_pool_size > 0)
            return env_importance.pool_sample(env_pool, env_pool_size, rng);
        return env_importance.sample_direction(rng);
    }

    float get_total_weight() {
        return triangle_count > 0 ? cdf[triangle_count - 1] : 0.f;
    }

    bool has_emissive_triangles() {
        return get_total_weight() > 0.f;
    }

    bool has_lights() {
        return p_env > 0.f || has_emissive_triangles();
    }

    // Probability of choosing the environment over the triangles.
    float get_env_probability() {
        if (!merian_scene_has_env_light || !env_importance.is_valid())
            return 0.f;
        return has_emissive_triangles() ? p_env : select(p_env > 0.f, 1.f, 0.f);
    }

    // --- triangle selection ---

    // Cell holding pos, or LIGHT_INVALID_INDEX outside the grid.
    uint get_grid_cell(const float3 pos) {
        if (selection != LightSelection::LightSelectionGrid)
            return LIGHT_INVALID_INDEX;
        const LightGridInfo info = grid_info[0];
        if (info.dimension == 0)
            return LIGHT_INVALID_INDEX;
        const int3 coord = int3(floor((pos - info.origin) * info.inv_cell_size));
        if (any(coord < 0) || any(coord >= int(info.dimension)))
            return LIGHT_INVALID_INDEX;
        return uint(coord.x) + info.dimension * (uint(coord.y) + info.dimension * uint(coord.z));
    }

    // Slots holding this light, and slots holding nothing, over the slot count. The first is the
    // density of a cell sample, the second the mass a cell hands back to the pool.
    void scan_cell(const uint cell, const uint index, out float p_cell, out float p_fallback) {
        uint count = 0;
        uint empty = 0;
        for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
            const uint slot_index = grid[cell * LIGHT_GRID_SLOTS + slot];
            count += uint(slot_index == index);
            empty += uint(slot_index == LIGHT_INVALID_INDEX);
        }
        p_cell = float(count) * (1.f / float(LIGHT_GRID_SLOTS));
        p_fallback = float(empty) * (1.f / float(LIGHT_GRID_SLOTS));
    }

    // Probability of the global technique picking a light. Taken from the stored prefix sums,
    // not from the weight: with many lights the difference of two neighbouring sums is what the
    // search actually realizes, and the two disagree once the sums lose the weight in rounding.
    float get_global_select_pdf(const uint index) {
        const float total = get_total_weight();
        if (!(total > 0.f))
            return 0.f;
        const float gap = index == 0 ? cdf[0] : cdf[index] - cdf[index - 1];
        return max(gap, 0.f) / total;
    }

    // First entry of the tile a shading point draws from. Which tile it is does not change the
    // density: every entry is an independent draw from the same distribution.
    uint pool_offset(const uint bucket) {
        if (pool_size <= pool_tile_size || bucket == LIGHT_INVALID_INDEX)
            return 0;
        const uint tiles = pool_size / pool_tile_size;
        return (bucket % tiles) * pool_tile_size;
    }

    // Chance of consulting the cell list at pos.
    float get_grid_probability(const uint cell) {
        return cell != LIGHT_INVALID_INDEX ? grid_probability : 0.f;
    }

    // Density with which select_light would pick this light in that cell: a mixture of the cell
    // list and the global pool, so it is never zero for a light that can contribute.
    float get_select_pdf(const uint cell, const uint index) {
        const float p_grid = get_grid_probability(cell);
        if (p_grid <= 0.f)
            return get_global_select_pdf(index);

        float p_cell, p_fallback;
        scan_cell(cell, index, p_cell, p_fallback);
        return p_grid * p_cell +
               (1.f - p_grid + p_grid * p_fallback) * get_global_select_pdf(index);
    }

    // One load from the cell list, or from the per-frame pool. Only the power technique still
    // searches the whole scene.
    uint select_light(const uint cell, inout RandomGenerator rng, out float p_select) {
        p_select = 0.f;
        if (!(get_total_weight() > 0.f))
            return LIGHT_INVALID_INDEX;

        const float p_grid = get_grid_probability(cell);
        uint index = LIGHT_INVALID_INDEX;

        if (p_grid > 0.f && rng.next_float() < p_grid) {
            const uint slot = min(uint(rng.next_float() * LIGHT_GRID_SLOTS), LIGHT_GRID_SLOTS - 1);
            index = grid[cell * LIGHT_GRID_SLOTS + slot];
        }
        // an empty slot hands the sample back to the global technique, which get_select_pdf
        // charges for
        if (index == LIGHT_INVALID_INDEX) {
            if (selection == LightSelection::LightSelectionPower || pool_size == 0) {
                const float target = rng.next_float() * cdf[triangle_count - 1];
                uint lo = 0;
                uint hi = triangle_count - 1;
                while (lo < hi) {
                    const uint mid = (lo + hi) >> 1;
                    if (cdf[mid] > target)
                        hi = mid;
                    else
                        lo = mid + 1;
                }
                index = lo;
            } else {
                index = pool[pool_offset(cell) +
                             min(uint(rng.next_float() * pool_tile_size), pool_tile_size - 1)];
                if (index == LIGHT_INVALID_INDEX)
                    return LIGHT_INVALID_INDEX;
            }
        }

        p_select = get_select_pdf(cell, index);
        return index;
    }

    Optional<uint> get_triangle_index(const GeometryID geometry_id, const uint primitive_id) {
        if (geometry_id >= geometry_count)
            return none;
        const uint offset = geometry_light_offsets[geometry_id];
        if (offset == LIGHT_INVALID_INDEX)
            return none;
        return offset + primitive_id;
    }

    // --- triangle sampling ---

    // Uniform point on triangle index; Li is the material emission at the point.
    Optional<LightSample> sample_triangle(const uint index,
                                          const float3 pos,
                                          inout RandomGenerator rng,
                                          const MaterialSystem material_system,
                                          const float time) {
        const EmissiveTriangle tri = triangles[index];
        const float3 bary = merian::sample_triangle(rng);
        const float3 light_pos = tri.get_position(bary);

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

        const ShadingData sd = ShadingData(-wo, light_pos, tri.get_uv(bary), time,
                                           OrthonormalFrame(n), n, uint16_t(1), 1.0f, front_facing);

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
        const EmissiveTriangle tri = triangles[index];
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

    // --- combined ---

    // Uniform point on a flux-selected light, without evaluating its emission.
    Optional<LightPoint>
    sample_point(const uint cell, const float3 pos, inout RandomGenerator rng) {
        const float p_e = get_env_probability();
        if (p_e <= 0.f && !has_emissive_triangles())
            return none;

        LightPoint lp;
        if (rng.next_float() < p_e) {
            if (let es = sample_env(rng)) {
                lp.pos = pos + es.dir * ENV_MAP_DISTANCE;
                lp.index = LIGHT_ENV_INDEX;
                return lp;
            }
            return none;
        }

        float p_select;
        const uint index = select_light(cell, rng, p_select);
        if (index == LIGHT_INVALID_INDEX || p_select <= 0.f)
            return none;

        const EmissiveTriangle tri = triangles[index];
        const float3 scaled_n = tri.get_scaled_normal();
        const float area2 = length(scaled_n);
        if (area2 <= 0.f)
            return none;
        const float3 light_pos = tri.get_position(merian::sample_triangle(rng));
        // the emitting side must face the receiver unless the triangle emits both ways
        if (dot(scaled_n, pos - light_pos) <= 0.f && !tri.is_two_sided())
            return none;
        lp.pos = light_pos;
        lp.index = index;
        return lp;
    }

    Optional<LightSample> sample<Env : IEnvMap>(const uint cell,
                                                const float3 pos,
                                                inout RandomGenerator rng,
                                                const MaterialSystem material_system,
                                                const Env env,
                                                const float time) {
        const float p_e = get_env_probability();
        if (p_e <= 0.f && !has_emissive_triangles())
            return none;

        if (rng.next_float() < p_e) {
            if (let es = sample_env(rng)) {
                LightSample ls;
                ls.wo = es.dir;
                ls.dist = SCENE_RAY_TMAX;
                ls.Li = env.eval(make_env_shading_data(es.dir, pos, time),
                                 material_system.texture_manager, DefaultLODSampler());
                ls.pdf = es.pdf * p_e;
                ls.pos = pos + es.dir * ENV_MAP_DISTANCE;
                ls.index = LIGHT_ENV_INDEX;
                return ls;
            }
            return none;
        }

        float p_select;
        const uint index = select_light(cell, rng, p_select);
        if (index == LIGHT_INVALID_INDEX || p_select <= 0.f)
            return none;
        const Optional<LightSample> opt = sample_triangle(index, pos, rng, material_system, time);
        if (!opt.hasValue)
            return none;
        LightSample ls = opt.value;
        ls.pdf *= (1.f - p_e) * p_select;
        return ls;
    }

    // Density of sample() producing the direction dir towards the environment.
    float pdf_env(const float3 dir) {
        if (!merian_scene_has_env_light)
            return 0.f;
        const float p_e = get_env_probability();
        return p_e > 0.f ? p_e * env_importance.pdf(dir) : 0.f;
    }

    // Density of sample() producing the point where the ray meets the triangle index. The
    // selection depends on where the ray started, so the density does too.
    float pdf_index(const uint cell, const uint index, const float3 origin, const float3 dir) {
        return (1.f - get_env_probability()) * get_select_pdf(cell, index) *
               pdf_triangle(index, origin, dir);
    }

    // The light-sampling side of a hit on the given primitive.
    LightHit get_light_hit(const uint cell,
                           const GeometryID geometry_id,
                           const uint primitive_id,
                           const float3 origin,
                           const float3 dir) {
        LightHit lh;
        if (let index = get_triangle_index(geometry_id, primitive_id)) {
            lh.index = index;
            lh.pdf = pdf_index(cell, index, origin, dir);
        }
        return lh;
    }
}

// Weighted reservoir over the direct-light candidates of one path vertex: light samples and the
// scatter continuation where it hit an emitter. Resampling weight p_hat / (M p_light + p_dir)
// (generalized balance heuristic, Bitterli et al. 2020), with p_hat = luminance of the unshadowed
// contribution; the estimate is value * weight_sum / p_hat times the visibility of the winner.
// Only what the shadow ray needs is kept: the value and the point to aim at.
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

    [mutating] void add(const float3 candidate_value,
                        const float pdf_denominator,
                        const uint index,
                        const float3 pos,
                        inout RandomGenerator rng) {
        const float weight = yuv_luminance(candidate_value) / pdf_denominator;
        if (!(weight > 0.f))
            return;
        weight_sum += weight;
        if (rng.next_float() * weight_sum < weight) {
            value = candidate_value;
            light_pos = pos;
            light_index = index;
        }
    }

        [mutating] void add_light(const LightSample ls,
                                  const float3 candidate_value,
                                  const float pdf_denominator,
                                  inout RandomGenerator rng) {
        add(candidate_value, pdf_denominator, ls.index, ls.pos, rng);
    }

    [mutating] void add_continuation(const float3 candidate_value,
                                     const float pdf_denominator,
                                     inout RandomGenerator rng) {
        add(candidate_value, pdf_denominator, LIGHT_INVALID_INDEX, float3(0), rng);
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

} // namespace merian
