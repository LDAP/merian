// Layered OpenPBR-style surface. The C++ side mirrors this struct byte-for-byte.

import merian_shaders.shading.materials.material;
import merian_shaders.shading.bsdfs.brdf_ggx;
import merian_shaders.shading.bsdfs.brdf_lambert_diffuse;
import merian_shaders.shading.bsdfs.brdf_sheen;
import merian_shaders.shading.bsdfs.bsdf_conductor_fresnel;
import merian_shaders.shading.bsdfs.bsdf_fresnel_mix;
import merian_shaders.shading.bsdfs.bsdf_mix;
import merian_shaders.utils.normal_map;

namespace merian {

typealias PBRTDielectric = FresnelMixBSDF<GGXBRDF, LambertDiffuseBRDF>; // specular over diffuse
typealias PBRTMetal = ConductorFresnelBSDF<GGXBRDF>;
typealias PBRTSubstrate = MixBSDF<PBRTDielectric, PBRTMetal>; // by metalness
typealias PBRTCoated = FresnelMixBSDF<GGXBRDF, PBRTSubstrate>;
typealias PBRTWithCoat = MixBSDF<PBRTSubstrate, PBRTCoated>; // by coat weight
typealias PBRTBSDF = MixBSDF<PBRTWithCoat, SheenBRDF>;       // by sheen weight

struct PBRTMaterial : MaterialModel {
    typealias MaterialSampleImpl = PBRTMaterialSample;

    MaterialHeader header; // alpha_texture_id = base color texture
    float3 base_color;
    float opacity;
    float metalness;
    float roughness;
    float specular_weight;
    float specular_ior;
    float3 emission;
    float normal_scale;
    float coat_weight;
    float coat_roughness;
    float coat_ior;
    float sheen_weight;
    float3 sheen_color;
    float sheen_roughness;
    TextureID metalness_texture;
    TextureID roughness_texture;
    TextureID emission_texture;
    TextureID normal_texture;

    PBRTMaterialSample
    get_sample(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        float3 base = base_color;
        if (let tex = header.get_alpha_texture_id()) {
            base *= lod.sample(tm.get_sampler2D(tex), sd.uv).rgb;
        }

        float metal = metalness;
        if (metalness_texture != TextureID(-1)) {
            metal *= lod.sample(tm.get_sampler2D(metalness_texture), sd.uv).r;
        }
        float rough = roughness;
        if (roughness_texture != TextureID(-1)) {
            rough *= lod.sample(tm.get_sampler2D(roughness_texture), sd.uv).r;
        }

        OrthonormalFrame shading_frame = sd.frame;
        if (normal_texture != TextureID(-1)) {
            const float3 raw = lod.sample(tm.get_sampler2D(normal_texture), sd.uv).rgb;
            const float2 nxy = (raw.xy * 2.0f - 1.0f) * normal_scale;
            const float nz = sqrt(max(0.0f, 1.0f - dot(nxy, nxy)));
            const OrthonormalFrame mapped = sd.frame.apply_normal_map(float3(nxy, nz));
            if (merian_hint_clamp_normals) {
                const float3 clamped_n =
                    keller_clamp_normal(mapped.get_normal(), sd.face_normal, sd.wi);
                shading_frame = OrthonormalFrame(clamped_n, mapped.get_tangent());
            } else {
                shading_frame = mapped;
            }
        }

        float3 emission_out = emission;
        if (emission_texture != TextureID(-1)) {
            emission_out *= lod.sample(tm.get_sampler2D(emission_texture), sd.uv).rgb;
        }

        const float alpha = ggx_roughness_to_alpha(max(rough, merian_hint_min_roughness));

        // Dielectric specular F0 from IOR, scaled by the specular weight.
        const float ior_t = (specular_ior - 1.0f) / (specular_ior + 1.0f);
        const float dielectric_f0 = clamp(specular_weight * ior_t * ior_t, 0.0f, 1.0f);

        const PBRTSubstrate substrate =
            PBRTSubstrate(PBRTDielectric(GGXBRDF(alpha), LambertDiffuseBRDF(base), dielectric_f0),
                          PBRTMetal(GGXBRDF(alpha), base), metal);

        // Dielectric clearcoat over the substrate.
        const float coat_alpha =
            ggx_roughness_to_alpha(max(coat_roughness, merian_hint_min_roughness));
        const float coat_t = (coat_ior - 1.0f) / (coat_ior + 1.0f);
        const PBRTWithCoat with_coat = PBRTWithCoat(
            substrate, PBRTCoated(GGXBRDF(coat_alpha), substrate, coat_t * coat_t), coat_weight);

        PBRTMaterialSample s;
        s.brdf = PBRTBSDF(with_coat, SheenBRDF(sheen_color, sheen_roughness), sheen_weight);
        s.shading_frame = shading_frame;
        s.emission = emission_out;
        return s;
    }
}

struct PBRTMaterialSample : MaterialSample {
    typealias BSDFImpl = PBRTBSDF;

    PBRTBSDF brdf;
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
