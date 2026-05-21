// glTF 2.0 metallic-roughness material model. The C++ side mirrors this struct byte-for-byte.

import merian_shaders.shading.materials.material;
import merian_shaders.shading.bsdfs.brdf_ggx;
import merian_shaders.shading.bsdfs.brdf_lambert_diffuse;
import merian_shaders.shading.bsdfs.bsdf_conductor_fresnel;
import merian_shaders.shading.bsdfs.bsdf_fresnel_mix;
import merian_shaders.shading.bsdfs.bsdf_mix;

namespace merian {

// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#complete-model
typealias GltfBSDF =
    MixBSDF<FresnelMixBSDF<GGXBRDF, LambertDiffuseBRDF>, ConductorFresnelBSDF<GGXBRDF>>;

static const float GLTF_DIELECTRIC_IOR_F0 = 0.04f; // equals ior = 1.5

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
        float4 base = base_color_factor;
        if (let tex = header.get_alpha_texture_id()) {
            base *= lod.sample(tm.get_sampler2D(tex), sd.uv);
        }

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

        float3 emission = emissive_factor;
        if (emissive_texture != TextureID(-1)) {
            emission *= lod.sample(tm.get_sampler2D(emissive_texture), sd.uv).rgb;
        }

        float occlusion = 1.0f;
        if (occlusion_texture != TextureID(-1)) {
            const float o = lod.sample(tm.get_sampler2D(occlusion_texture), sd.uv).r;
            occlusion = lerp(1.0f, o, occlusion_strength);
        }

        const float alpha = ggx_roughness_to_alpha(roughness);

        GltfMaterialSample s;
        s.brdf = GltfBSDF(
            FresnelMixBSDF<GGXBRDF, LambertDiffuseBRDF>(
                GGXBRDF(alpha), LambertDiffuseBRDF(base.rgb * occlusion), GLTF_DIELECTRIC_IOR_F0),
            ConductorFresnelBSDF<GGXBRDF>(GGXBRDF(alpha), base.rgb), metallic);
        s.shading_frame = shading_frame;
        s.emission = emission;
        return s;
    }
}

struct GltfMaterialSample : MaterialSample {
    typealias BSDFImpl = GltfBSDF;

    GltfBSDF brdf;
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
