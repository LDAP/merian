#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.utils.textures;
import merian_shaders.utils.envmap;
import merian_shaders.utils.math;
__exported import merian_shaders.shading.materials.material;

namespace merian {

// virtual hit point distance
static const float ENV_MAP_DISTANCE = 1e7f;

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

    static float2 dir_to_uv(const float3 d) {
        const float3 a = abs(d);
        const float r = sqrt(max(0.f, 1.f - a.z));
        const float m = max(a.x, a.y);
        const float b = m == 0.f ? 0.f : min(a.x, a.y) / m;
        float phi = atan(b) * (2.f / MERIAN_M_PI);
        if (a.x < a.y) {
            phi = 1.f - phi;
        }
        float v = phi * r;
        float u = r - v;
        if (d.z < 0.f) {
            const float mirrored_u = u;
            u = 1.f - v;
            v = 1.f - mirrored_u;
        }
        u = d.x < 0.f ? -u : u;
        v = d.y < 0.f ? -v : v;
        return 0.5f * float2(u + 1.f, v + 1.f);
    }

    float3 eval(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        return intensity * lod.sample(env_map, dir_to_uv(mul((float3x3)to_local, -sd.wi))).rgb;
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

}
