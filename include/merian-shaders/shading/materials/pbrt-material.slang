// Layered OpenPBR-style surface. Generic over the optional lobes it carries; an absent lobe is a
// Conditional<T, false> member that occupies no payload and folds out of get_sample. The C++ side
// mirrors the present members byte-for-byte.

import merian_shaders.shading.materials.material;
import merian_shaders.shading.bsdfs.brdf_ggx;
import merian_shaders.shading.bsdfs.brdf_lambert_diffuse;
import merian_shaders.shading.bsdfs.brdf_sheen;
import merian_shaders.shading.bsdfs.bsdf_conductor_fresnel;
import merian_shaders.shading.bsdfs.bsdf_fresnel_mix;
import merian_shaders.shading.bsdfs.bsdf_mix;
import merian_shaders.shading.bsdfs.bsdf_rough_dielectric;
import merian_shaders.utils.normal_map;

namespace merian {

// Opaque part: specular reflection over diffuse. Transmissive part: a coupled rough dielectric
// (TIR-aware). Mixed by the transmission weight.
typealias PBRTOpaqueDielectric = FresnelMixBSDF<SpecularBRDF, LambertDiffuseBRDF>;
typealias PBRTDielectric = MixBSDF<PBRTOpaqueDielectric, RoughDielectricBSDF>;
typealias PBRTMetal = ConductorFresnelBSDF<SpecularBRDF>;
typealias PBRTSubstrate = MixBSDF<PBRTDielectric, PBRTMetal>; // by metalness
typealias PBRTCoated = FresnelMixBSDF<SpecularBRDF, PBRTSubstrate>;
typealias PBRTWithCoat = MixBSDF<PBRTSubstrate, PBRTCoated>; // by coat weight
typealias PBRTBSDF = MixBSDF<PBRTWithCoat, SheenBRDF>;       // by sheen weight

struct PBRTTransmission {
    float weight;
    float3 color;
};
struct PBRTClearcoat {
    float weight;
    float roughness;
    float ior;
};
struct PBRTSheen {
    float weight;
    float3 color;
    float roughness;
};

struct PBRTMaterial<let TRANSMISSION : bool = false,
                    let CLEARCOAT : bool = false,
                    let SHEEN : bool = false> : MaterialModel {
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
    TextureID metalness_texture;
    TextureID roughness_texture;
    TextureID emission_texture;
    TextureID normal_texture;

    Conditional<PBRTTransmission, TRANSMISSION> transmission;
    Conditional<PBRTClearcoat, CLEARCOAT> clearcoat;
    Conditional<PBRTSheen, SHEEN> sheen;

    property Optional<PBRTTransmission> transmission_opt {
        get {
            return transmission;
        }
    }
    property Optional<PBRTClearcoat> clearcoat_opt {
        get {
            return clearcoat;
        }
    }
    property Optional<PBRTSheen> sheen_opt {
        get {
            return sheen;
        }
    }

    PBRTMaterialSample
    get_sample(const ShadingData sd, const TextureManager tm, const LODSampler2D lod) {
        const LOD0Sampler lod0;

        float3 base = base_color;
        if (let tex = header.get_alpha_texture_id()) {
            base *= lod.sample(tm.get_sampler2D(tex), sd.uv).rgb;
        }

        float metal = metalness;
        if (metalness_texture != TextureID(-1)) {
            metal *= lod0.sample(tm.get_sampler2D(metalness_texture), sd.uv).r;
        }
        float rough = roughness;
        if (roughness_texture != TextureID(-1)) {
            rough *= lod0.sample(tm.get_sampler2D(roughness_texture), sd.uv).r;
        }

        OrthonormalFrame shading_frame = sd.frame;
        if (normal_texture != TextureID(-1)) {
            const float3 raw = lod0.sample(tm.get_sampler2D(normal_texture), sd.uv).rgb;
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

        float eta = 1.0f;
        float tw = 0.0f;
        float3 transmission_color = float3(1.0f);
        if (let trans = transmission_opt) {
            eta = sd.ior / (sd.front_facing ? specular_ior : 1.0f);
            tw = trans.weight;
            transmission_color = trans.color;
        }

        const PBRTDielectric dielectric = PBRTDielectric(
            PBRTOpaqueDielectric(SpecularBRDF(alpha), LambertDiffuseBRDF(base), dielectric_f0),
            RoughDielectricBSDF(alpha, eta, transmission_color), tw);
        const PBRTSubstrate substrate =
            PBRTSubstrate(dielectric, PBRTMetal(SpecularBRDF(alpha), base), metal);

        float cw = 0.0f;
        float coat_alpha = ggx_roughness_to_alpha(merian_hint_min_roughness);
        float coat_f0 = 0.0f;
        if (let coat = clearcoat_opt) {
            cw = coat.weight;
            coat_alpha = ggx_roughness_to_alpha(max(coat.roughness, merian_hint_min_roughness));
            const float coat_t = (coat.ior - 1.0f) / (coat.ior + 1.0f);
            coat_f0 = coat_t * coat_t;
        }
        const PBRTWithCoat with_coat =
            PBRTWithCoat(substrate, PBRTCoated(SpecularBRDF(coat_alpha), substrate, coat_f0), cw);

        float sw = 0.0f;
        float3 sheen_col = float3(0.0f);
        float sheen_rough = 0.0f;
        if (let sh = sheen_opt) {
            sw = sh.weight;
            sheen_col = sh.color;
            sheen_rough = sh.roughness;
        }

        PBRTMaterialSample s;
        s.brdf = PBRTBSDF(with_coat, SheenBRDF(sheen_col, sheen_rough), sw);
        s.shading_frame = shading_frame;
        s.emission = emission_out;
        s.eta = eta;
        return s;
    }
}

struct PBRTMaterialSample : MaterialSample {
    typealias BSDFImpl = PBRTBSDF;

    PBRTBSDF brdf;
    OrthonormalFrame shading_frame;
    float3 emission;
    float eta;

    BSDFImpl get_bsdf() {
        return brdf;
    }

    property float3 shading_normal {
        get {
            return shading_frame.get_normal();
        }
    }

    override float get_eta() {
        return eta;
    }
}

} // namespace merian
