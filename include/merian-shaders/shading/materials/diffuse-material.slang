import merian_shaders.shading.materials.material;
import merian_shaders.shading.bsdfs.brdf_lambert_diffuse;

namespace merian {

// DiffuseMaterial: uses the alpha_texture in MaterialHeader as the base_color_texture.
struct DiffuseMaterial : MaterialModel {
    typealias MaterialSampleImpl = DiffuseMaterialSample;

    MaterialHeader header;
    float4 base_color_factor;

    DiffuseMaterialSample
    get_sample(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        float3 albedo = base_color_factor.rgb;
        if (let tex_id = header.get_alpha_texture_id()) {
            float4 tex_color = lod.sample(tm.get_sampler2D(tex_id), sd.uv);
            albedo *= tex_color.rgb;
        }
        DiffuseMaterialSample s;
        s.brdf = LambertDiffuseBRDF(albedo);
        s.frame = sd.frame.apply_normal_map(float3(0, 0, 1));
        s.emission = float3(0);
        return s;
    }

    static DiffuseMaterialSample make_emissive(const float3 emission, const float3 normal) {
        DiffuseMaterialSample s;
        s.brdf = LambertDiffuseBRDF(float3(0));
        s.frame = OrthonormalFrame(normal);
        s.emission = emission;
        return s;
    }

    static DiffuseMaterialSample
    unpack_bsdf(const uint2 packed, const float3 emission, const OrthonormalFrame frame) {
        DiffuseMaterialSample s;
        s.brdf = LambertDiffuseBRDF(float3(f16tof32(packed.x), f16tof32(packed.x >> 16),
                                           f16tof32(packed.y)));
        s.frame = frame;
        s.emission = emission;
        return s;
    }
}

struct DiffuseMaterialSample : MaterialSample {
    typealias BSDFImpl = LambertDiffuseBRDF;
    LambertDiffuseBRDF brdf;
    OrthonormalFrame frame;
    float3 emission;
    BSDFImpl get_bsdf() {
        return brdf;
    }
    property OrthonormalFrame shading_frame { get { return frame; } }
    property float3 shading_normal { get { return frame.get_normal(); } }
    override uint2 pack_bsdf() {
        return uint2(f32tof16(brdf.albedo.x) | (f32tof16(brdf.albedo.y) << 16),
                     f32tof16(brdf.albedo.z));
    }
}

}
