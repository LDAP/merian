#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.utils.textures;
import merian_shaders.utils.envmap;
import merian_shaders.utils.math;
import merian_shaders.utils.sampling;
__exported import merian_shaders.shading.materials.material;

namespace merian {

// virtual hit point distance
static const float ENV_MAP_DISTANCE = 1e7f;

// ShadingData for looking up the environment along dir (pointing away from the receiver).
ShadingData make_env_shading_data(const float3 dir, const float3 origin, const float time) {
    return ShadingData(-dir, origin + dir * ENV_MAP_DISTANCE, float2(0), time,
                       OrthonormalFrame(-dir), -dir, uint16_t(1), true);
}

struct EnvSample {
    float3 dir; // towards the environment
    float pdf;  // solid angle
    float3 Le;  // radiance arriving from dir
}

interface IEnvMap {
    float3 eval(const ShadingData sd, const TextureManager tm, const LODSampler2D lod);
}

struct EmptyEnv : IEnvMap {
    float3 eval(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        return float3(0);
    }
}

struct LatLongMap : IEnvMap {
    Sampler2D<float4> env_map;
    float4x4 to_local;
    float intensity;

    float3 eval(const float2 uv, const LODSampler2D lod) {
        return intensity * lod.sample(env_map, uv).rgb;
    }

    float3 eval(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        return eval(world_to_latlong_map(mul((float3x3)to_local, -sd.wi)), lod);
    }
}

// Equal-area octahedral square parameterization (Clarberg 2008), as used by pbrt-v4 infinite
// lights. Mapping derived from pbrt-v4 EqualAreaSphereToSquare, Apache-2.0.
struct EqualAreaOctMap : IEnvMap {
    Sampler2D<float4> env_map;
    float4x4 to_local;
    float intensity;

    float3 eval(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        return intensity *
               lod.sample(env_map, equal_area_sphere_to_square(mul((float3x3)to_local, -sd.wi)))
                   .rgb;
    }
}

struct CubeMap : IEnvMap {
    Sampler2D<float4> rt;
    Sampler2D<float4> bk;
    Sampler2D<float4> lf;
    Sampler2D<float4> ft;
    Sampler2D<float4> up;
    Sampler2D<float4> dn;
    float4x4 to_local;
    float intensity;

    float3 eval(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        const float3 d = mul((float3x3)to_local, -sd.wi);

        float3 c;
        switch (cubemap_side(d)) {
        case 0:
            c = lod.sample(rt, 0.5 + 0.5 * float2(-d.y, -d.z) / abs(d.x)).rgb;
            break;
        case 1:
            c = lod.sample(lf, 0.5 + 0.5 * float2(d.y, -d.z) / abs(d.x)).rgb;
            break;
        case 2:
            c = lod.sample(bk, 0.5 + 0.5 * float2(d.x, -d.z) / abs(d.y)).rgb;
            break;
        case 3:
            c = lod.sample(ft, 0.5 + 0.5 * float2(-d.x, -d.z) / abs(d.y)).rgb;
            break;
        case 4:
            c = lod.sample(up, 0.5 + 0.5 * float2(-d.y, d.x) / abs(d.z)).rgb;
            break;
        default:
            c = lod.sample(dn, 0.5 + 0.5 * float2(-d.y, -d.x) / abs(d.z)).rgb;
            break;
        }
        return intensity * c;
    }
}

} // namespace merian
