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
        return s;
    }
}

struct DiffuseMaterialSample : MaterialSample {
    typealias BSDFImpl = LambertDiffuseBRDF;
    LambertDiffuseBRDF brdf;
    OrthonormalFrame frame;
    BSDFImpl get_bsdf() {
        return brdf;
    }
    property float3 emission { get { return float3(0); } }
    property OrthonormalFrame shading_frame { get { return frame; } }
    property float3 shading_normal { get { return frame.get_normal(); } }
}

}
