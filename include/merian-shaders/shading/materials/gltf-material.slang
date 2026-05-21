// glTF 2.0 metallic-roughness material model. The C++ side mirrors this struct byte-for-byte.

import merian_shaders.shading.materials.material;
import merian_shaders.shading.bsdfs.brdf_lambert_ggx;
import merian_shaders.shading.bsdfs.brdf_ggx;

namespace merian {

struct GltfMaterial : MaterialModel {
    typealias MaterialSampleImpl = GltfMaterialSample;

    MaterialHeader header; // alpha_texture_id = baseColor texture
    float4 base_color_factor;
    float3 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    float occlusion_strength;
    TextureID metallic_roughness_texture;
    TextureID normal_texture;
    TextureID occlusion_texture;
    TextureID emissive_texture;

    GltfMaterialSample
    get_sample(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        // base color (sRGB texture × linear factor)
        float4 base = base_color_factor;
        if (let tex = header.get_alpha_texture_id()) {
            base *= lod.sample(tm.get_sampler2D(tex), sd.uv);
        }

        // metallic-roughness (linear; glTF: B=metallic, G=roughness)
        float metallic = metallic_factor;
        float roughness = roughness_factor;
        if (metallic_roughness_texture != TextureID(-1)) {
            const float4 mr = lod.sample(tm.get_sampler2D(metallic_roughness_texture), sd.uv);
            roughness *= mr.g;
            metallic *= mr.b;
        }

        // normal map per glTF: xy = (2*sample - 1) * normal_scale; z reconstructed
        OrthonormalFrame shading_frame = sd.frame;
        if (normal_texture != TextureID(-1)) {
            const float3 raw = lod.sample(tm.get_sampler2D(normal_texture), sd.uv).rgb;
            const float2 nxy = (raw.xy * 2.0f - 1.0f) * normal_scale;
            const float nz = sqrt(max(0.0f, 1.0f - dot(nxy, nxy)));
            shading_frame = sd.frame.apply_normal_map(float3(nxy, nz));
        }

        // emissive (sRGB texture × linear factor)
        float3 emission = emissive_factor;
        if (emissive_texture != TextureID(-1)) {
            emission *= lod.sample(tm.get_sampler2D(emissive_texture), sd.uv).rgb;
        }

        // occlusion (linear), folded into the diffuse lobe
        float occlusion = 1.0f;
        if (occlusion_texture != TextureID(-1)) {
            const float o = lod.sample(tm.get_sampler2D(occlusion_texture), sd.uv).r;
            occlusion = lerp(1.0f, o, occlusion_strength);
        }

        const float3 F0 = lerp(float3(0.04f), base.rgb, metallic);
        const float3 diffuse = base.rgb * (1.0f - metallic) * occlusion;
        const float alpha = ggx_roughness_to_alpha(roughness);

        GltfMaterialSample s;
        s.brdf = LambertGGXBRDF(diffuse, F0, alpha);
        s.shading_frame = shading_frame;
        s.emission = emission;
        return s;
    }
}

// Field names match the MaterialSample interface property names so Slang
// satisfies the property requirements from the fields directly.
struct GltfMaterialSample : MaterialSample {
    typealias BSDFImpl = LambertGGXBRDF;

    LambertGGXBRDF brdf;
    OrthonormalFrame shading_frame;
    float3 emission;

    BSDFImpl get_bsdf() {
        return brdf;
    }

    property float3 shading_normal {
        get { return shading_frame.get_normal(); }
    }
}

}
