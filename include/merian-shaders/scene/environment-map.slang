#pragma once

import merian_shaders.utils.textures;
import merian_shaders.utils.envmap;
import merian_shaders.utils.math;
__exported import merian_shaders.shading.materials.material;

namespace merian {

// virtual hit point distance
static const float ENV_MAP_DISTANCE = 1e7f;

interface IEnvMap {
    float3 eval(const float3 dir, const LODSampler2D lod);
}

struct EmptyEnv : IEnvMap {
    float3 eval(const float3 dir, const LODSampler2D lod) {
        return float3(0);
    }
}

struct LatLongMap : IEnvMap {
    Sampler2D env_map;
    float3x3 to_local;

    float3 eval(const float2 uv, const LODSampler2D lod) {
        return lod.sample(env_map, uv).rgb;
    }

    float3 eval(const float3 dir, const LODSampler2D lod) {
        return eval(world_to_latlong_map(mul(to_local, dir)), lod);
    }
}

struct CubeMap : IEnvMap {
    Sampler2D rt;
    Sampler2D bk;
    Sampler2D lf;
    Sampler2D ft;
    Sampler2D up;
    Sampler2D dn;
    float3x3 to_local;

    float3 eval(const float3 dir, const LODSampler2D lod) {
        const float3 d = mul(to_local, dir);
        Sampler2D tex = rt;
        float2 uv;

        switch (cubemap_side(d)) {
        case 0: {
            tex = rt;
            uv = 0.5 + 0.5 * float2(-d.y, -d.z) / abs(d.x);
            break;
        }
        case 1: {
            tex = lf;
            uv = 0.5 + 0.5 * float2(d.y, -d.z) / abs(d.x);
            break;
        }
        case 2: {
            tex = bk;
            uv = 0.5 + 0.5 * float2(d.x, -d.z) / abs(d.y);
            break;
        }
        case 3: {
            tex = ft;
            uv = 0.5 + 0.5 * float2(-d.x, -d.z) / abs(d.y);
            break;
        }
        case 4: {
            tex = up;
            uv = 0.5 + 0.5 * float2(-d.y, d.x) / abs(d.z);
            break;
        }
        case 5: {
            tex = dn;
            uv = 0.5 + 0.5 * float2(-d.y, -d.x) / abs(d.z);
            break;
        }
        }

        return lod.sample(tex, uv).rgb;
    }
}

struct EnvMaterialSample : MaterialSample {
    typealias BSDFImpl = NullBSDF;

    float3 _emission;
    OrthonormalFrame _frame;

    property float3 emission {
        get { return _emission; }
    }
    property OrthonormalFrame shading_frame {
        get { return _frame; }
    }
    property float3 shading_normal {
        get { return _frame.get_normal(); }
    }

    BSDFImpl get_bsdf() {
        return NullBSDF();
    }

    __init(const float3 env_emission, const float3 ray_direction) {
        _emission = env_emission;
        _frame = OrthonormalFrame(-ray_direction);
    }
}

}
