import merian_shaders.shading.materials.material;
import merian_shaders.shading.bsdfs.brdf_lambert_diffuse;

namespace merian {

// DiffuseMaterial: uses the alpha_texture in MaterialHeader as the base_color_texture.
struct DiffuseMaterial : MaterialModel {
    typealias MaterialSampleImpl = DiffuseMaterialSample;

    MaterialHeader header;
    float4 base_color_factor;

    DiffuseMaterialSample get_sample(const ShadingData sd,
                                     const TextureManager tm,
                                     const LODSampler2D lod,
                                     out float3 out_emission,
                                     out OrthonormalFrame out_shading_frame) {
        out_emission = float3(0);
        out_shading_frame = sd.frame.apply_normal_map(float3(0, 0, 1));
        float3 albedo = base_color_factor.rgb;
        if (let tex_id = header.get_alpha_texture_id()) {
            float4 tex_color = lod.sample(tm.get_sampler2D(tex_id), sd.uv);
            albedo *= tex_color.rgb;
        }
        DiffuseMaterialSample s;
        s.brdf = LambertDiffuseBRDF(albedo);
        return s;
    }
}

struct DiffuseMaterialSample : MaterialSample {
    typealias BSDFImpl = LambertDiffuseBRDF;
    LambertDiffuseBRDF brdf;
    BSDFImpl get_bsdf() {
        return brdf;
    }
}

}
