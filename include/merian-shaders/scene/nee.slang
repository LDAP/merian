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

// light sampling is wasted below this roughness: the BSDF is far more peaked than the light
static const float NEE_MIN_ROUGHNESS = 0.05f;

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
    // inclusive prefix sum of the triangle flux
    StructuredBuffer<float> cdf;
    // GeometryID -> first triangle index, LIGHT_INVALID_INDEX for non-emissive geometry
    StructuredBuffer<uint> geometry_light_offsets;
    uint triangle_count;
    uint geometry_count;
    // 0 if the environment is not sampled
    float p_env;
    // UseEnvMap geometry exists: rays towards the environment must find the closest hit
    bool has_sky_portals;
    EnvImportance env_importance;

    float get_total_flux() {
        return triangle_count > 0 ? cdf[triangle_count - 1] : 0.f;
    }

    bool has_emissive_triangles() {
        return get_total_flux() > 0.f;
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

    // Selects a triangle with probability flux / total flux (upper_bound over the cdf).
    uint select_triangle(const float u, out float p_select) {
        const float total = cdf[triangle_count - 1];
        const float target = u * total;
        uint lo = 0;
        uint hi = triangle_count - 1;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (cdf[mid] > target)
                hi = mid;
            else
                lo = mid + 1;
        }
        p_select = triangles[lo].flux / total;
        return lo;
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
    Optional<LightPoint> sample_point(const float3 pos, inout RandomGenerator rng) {
        const float p_e = get_env_probability();
        if (p_e <= 0.f && !has_emissive_triangles())
            return none;

        LightPoint lp;
        if (rng.next_float() < p_e) {
            if (let es = env_importance.sample_direction(rng)) {
                lp.pos = pos + es.dir * ENV_MAP_DISTANCE;
                lp.index = LIGHT_ENV_INDEX;
                return lp;
            }
            return none;
        }

        float p_select;
        const uint index = select_triangle(rng.next_float(), p_select);
        if (p_select <= 0.f)
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

    Optional<LightSample> sample<Env : IEnvMap>(const float3 pos,
                                                inout RandomGenerator rng,
                                                const MaterialSystem material_system,
                                                const Env env,
                                                const float time) {
        const float p_e = get_env_probability();
        if (p_e <= 0.f && !has_emissive_triangles())
            return none;

        if (rng.next_float() < p_e) {
            if (let es = env_importance.sample_direction(rng)) {
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
        const uint index = select_triangle(rng.next_float(), p_select);
        if (p_select <= 0.f)
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

    // Density of sample() producing the point where the ray meets the triangle index.
    float pdf_index(const uint index, const float3 origin, const float3 dir) {
        const float total = get_total_flux();
        if (total <= 0.f)
            return 0.f;
        return (1.f - get_env_probability()) * (triangles[index].flux / total) *
               pdf_triangle(index, origin, dir);
    }

    // The light-sampling side of a hit on the given primitive.
    LightHit get_light_hit(const GeometryID geometry_id,
                           const uint primitive_id,
                           const float3 origin,
                           const float3 dir) {
        LightHit lh;
        if (let index = get_triangle_index(geometry_id, primitive_id)) {
            lh.index = index;
            lh.pdf = pdf_index(index, origin, dir);
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
