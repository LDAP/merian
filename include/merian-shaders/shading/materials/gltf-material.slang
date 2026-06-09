// glTF 2.0 metallic-roughness material model. The C++ side mirrors this struct byte-for-byte.

import merian_shaders.shading.materials.material;
import merian_shaders.shading.bsdfs.brdf_ggx;
import merian_shaders.shading.bsdfs.brdf_lambert_diffuse;
import merian_shaders.shading.bsdfs.bsdf_conductor_fresnel;
import merian_shaders.shading.bsdfs.bsdf_fresnel_mix;
import merian_shaders.shading.bsdfs.bsdf_mix;
import merian_shaders.utils.normal_map;

namespace merian {

// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#complete-model
// KHR_materials_transmission: the transmission BTDF takes the diffuse slot, mixed by the transmission.
typealias GltfDiffuseOrTransmission = MixBSDF<LambertDiffuseBRDF, SpecularBTDF>;
typealias GltfDielectric = FresnelMixBSDF<SpecularBRDF, GltfDiffuseOrTransmission>;
typealias GltfReflective = MixBSDF<GltfDielectric, ConductorFresnelBSDF<SpecularBRDF>>;
// KHR_materials_clearcoat: a GGX coat over the surface, mixed by the clearcoat weight.
typealias GltfCoated = FresnelMixBSDF<SpecularBRDF, GltfReflective>;
typealias GltfBSDF = MixBSDF<GltfReflective, GltfCoated>;

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
    float transmission_weight;
    float ior;
    float clearcoat_weight;
    float clearcoat_roughness;
    float3 absorption; // volume absorption coefficient (sigma_a), precomputed by the loader
    TextureID metallic_roughness_texture;
    TextureID normal_texture;
    TextureID occlusion_texture;
    TextureID emissive_texture;
    TextureID clearcoat_texture;
    TextureID clearcoat_roughness_texture;

    float get_metallic(const TextureManager tm,
                       const float2 uv,
                       const LODSampler2D lod = DefaultLODSampler()) {
        float metallic = metallic_factor;
        if (metallic_roughness_texture != TextureID(-1)) {
            metallic *= lod.sample(tm.get_sampler2D(metallic_roughness_texture), uv).b;
        }
        return metallic;
    }

    float get_roughness(const TextureManager tm,
                        const float2 uv,
                        const LODSampler2D lod = DefaultLODSampler()) {
        float roughness = roughness_factor;
        if (metallic_roughness_texture != TextureID(-1)) {
            roughness *= lod.sample(tm.get_sampler2D(metallic_roughness_texture), uv).g;
        }
        return roughness;
    }

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
            const OrthonormalFrame mapped = sd.frame.apply_normal_map(float3(nxy, nz));
            if (merian_hint_clamp_normals) {
                const float3 clamped_n =
                    keller_clamp_normal(mapped.get_normal(), sd.face_normal, sd.wi);
                shading_frame = OrthonormalFrame(clamped_n, mapped.get_tangent());
            } else {
                shading_frame = mapped;
            }
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

        const float alpha = ggx_roughness_to_alpha(max(roughness, merian_hint_min_roughness));

        // Transmission weight and eta, gated by the toggle (off -> weight 0 -> the BTDF drops out).
        float eta = 1.0f;
        float tw = 0.0f;
        if (merian_hint_enable_transmission) {
            eta = sd.ior / (sd.front_facing ? ior : 1.0f);
            tw = transmission_weight;
        }
        const GltfDielectric dielectric = GltfDielectric(
            SpecularBRDF(alpha),
            GltfDiffuseOrTransmission(LambertDiffuseBRDF(base.rgb * occlusion),
                                      SpecularBTDF(alpha, eta, base.rgb), tw),
            GLTF_DIELECTRIC_IOR_F0);
        const GltfReflective reflective = GltfReflective(
            dielectric, ConductorFresnelBSDF<SpecularBRDF>(SpecularBRDF(alpha), base.rgb), metallic);

        // Clearcoat weight and roughness, gated by the toggle.
        float coat_weight = 0.0f;
        float coat_alpha = ggx_roughness_to_alpha(merian_hint_min_roughness);
        if (merian_hint_enable_clearcoat) {
            coat_weight = clearcoat_weight;
            if (clearcoat_texture != TextureID(-1)) {
                coat_weight *= lod.sample(tm.get_sampler2D(clearcoat_texture), sd.uv).r;
            }
            float coat_rough = clearcoat_roughness;
            if (clearcoat_roughness_texture != TextureID(-1)) {
                coat_rough *= lod.sample(tm.get_sampler2D(clearcoat_roughness_texture), sd.uv).g;
            }
            coat_alpha = ggx_roughness_to_alpha(max(coat_rough, merian_hint_min_roughness));
        }

        // Volume absorption of the medium entered on a front-face transmission; the integrator applies
        // it over the segment travelled inside.
        float3 absorb = float3(0.0f);
        if (merian_hint_enable_volume && sd.front_facing) {
            absorb = absorption;
        }

        GltfMaterialSample s;
        s.brdf = GltfBSDF(
            reflective, GltfCoated(SpecularBRDF(coat_alpha), reflective, GLTF_DIELECTRIC_IOR_F0),
            coat_weight);
        s.shading_frame = shading_frame;
        s.emission = emission;
        s.eta = eta;
        s.absorption = absorb;
        return s;
    }
}

struct GltfMaterialSample : MaterialSample {
    typealias BSDFImpl = GltfBSDF;

    GltfBSDF brdf;
    OrthonormalFrame shading_frame;
    float3 emission;
    float eta;
    float3 absorption;

    BSDFImpl get_bsdf() {
        return brdf;
    }

    property float3 shading_normal {
        get { return shading_frame.get_normal(); }
    }

    override float get_eta() {
        return eta;
    }

    override float3 get_absorption() {
        return absorption;
    }
}

}
