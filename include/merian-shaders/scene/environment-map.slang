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
    Sampler2D env_map;
    float4x4 to_local;

    float3 eval(const float2 uv, const LODSampler2D lod) {
        return lod.sample(env_map, uv).rgb;
    }

    float3 eval(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        return eval(world_to_latlong_map(mul((float3x3)to_local, -sd.wi)), lod);
    }
}

struct CubeMap : IEnvMap {
    Sampler2D rt;
    Sampler2D bk;
    Sampler2D lf;
    Sampler2D ft;
    Sampler2D up;
    Sampler2D dn;
    float4x4 to_local;

    float3 eval(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        const float3 d = mul((float3x3)to_local, -sd.wi);

        switch (cubemap_side(d)) {
        case 0:
            return lod.sample(rt, 0.5 + 0.5 * float2(-d.y, -d.z) / abs(d.x)).rgb;
        case 1:
            return lod.sample(lf, 0.5 + 0.5 * float2(d.y, -d.z) / abs(d.x)).rgb;
        case 2:
            return lod.sample(bk, 0.5 + 0.5 * float2(d.x, -d.z) / abs(d.y)).rgb;
        case 3:
            return lod.sample(ft, 0.5 + 0.5 * float2(-d.x, -d.z) / abs(d.y)).rgb;
        case 4:
            return lod.sample(up, 0.5 + 0.5 * float2(-d.y, d.x) / abs(d.z)).rgb;
        default:
            return lod.sample(dn, 0.5 + 0.5 * float2(-d.y, -d.x) / abs(d.z)).rgb;
        }
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

__generic<E : IEnvMap>
extension E : merian::MaterialModel {
    typealias MaterialSampleImpl = EnvMaterialSample;

    EnvMaterialSample get_sample(const ShadingData sd,
                                 const TextureManager tm,
                                 const LODSampler2D lod) {
        return EnvMaterialSample(eval(sd, tm, lod), -sd.wi);
    }
}

}
