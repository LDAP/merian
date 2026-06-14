// glTF 2.0 metallic-roughness material model. The C++ side mirrors this struct byte-for-byte.

import merian_shaders.shading.materials.material;
import merian_shaders.shading.bsdfs.brdf_ggx;
import merian_shaders.shading.bsdfs.brdf_lambert_diffuse;
import merian_shaders.shading.bsdfs.brdf_sheen;
import merian_shaders.shading.bsdfs.bsdf_conductor_fresnel;
import merian_shaders.shading.bsdfs.bsdf_fresnel_mix;
import merian_shaders.shading.bsdfs.bsdf_iridescence;
import merian_shaders.shading.bsdfs.bsdf_mix;
import merian_shaders.shading.bsdfs.bsdf_sheen_layer;
import merian_shaders.utils.normal_map;

namespace merian {

// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#complete-model

// Opaque dielectric: specular over diffuse, Fresnel modulated by KHR_materials_iridescence.
typealias GltfOpaqueDielectric = IridescentFresnelMixBSDF<SpecularBRDF, LambertDiffuseBRDF>;
// KHR_materials_transmission: coupled iridescent rough dielectric, mixed by the weight.
typealias GltfDielectric = MixBSDF<GltfOpaqueDielectric, IridescentRoughDielectricBSDF>;
typealias GltfConductor = IridescentConductorBSDF<SpecularBRDF>;
typealias GltfReflective = MixBSDF<GltfDielectric, GltfConductor>;
// KHR_materials_sheen: sheen lobe over the reflective base.
typealias GltfSheened = SheenLayerBSDF<GltfReflective>;
// KHR_materials_clearcoat: a GGX coat over the sheened surface, mixed by the clearcoat weight.
typealias GltfCoated = FresnelMixBSDF<SpecularBRDF, GltfSheened>;
typealias GltfBSDF = MixBSDF<GltfSheened, GltfCoated>;

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
    float3 sheen_color;
    float sheen_roughness;
    float iridescence_factor;
    float iridescence_ior;
    float iridescence_thickness_min;
    float iridescence_thickness_max;
    float anisotropy_strength;
    float anisotropy_rotation;
    TextureID metallic_roughness_texture;
    TextureID normal_texture;
    TextureID occlusion_texture;
    TextureID emissive_texture;
    TextureID clearcoat_texture;
    TextureID clearcoat_roughness_texture;
    TextureID sheen_color_texture;
    TextureID sheen_roughness_texture;
    TextureID iridescence_texture;
    TextureID iridescence_thickness_texture;
    TextureID anisotropy_texture;
    TextureID _pad;

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
        const LOD0Sampler lod0;

        float4 base = base_color_factor;
        if (let tex = header.get_alpha_texture_id()) {
            base *= lod.sample(tm.get_sampler2D(tex), sd.uv);
        }

        float metallic = metallic_factor;
        float roughness = roughness_factor;
        if (metallic_roughness_texture != TextureID(-1)) {
            const float4 mr = lod0.sample(tm.get_sampler2D(metallic_roughness_texture), sd.uv);
            roughness *= mr.g;
            metallic *= mr.b;
        }

        // normal map per glTF: xy = (2*sample - 1) * normal_scale; z reconstructed
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

        float3 emission = emissive_factor;
        if (emissive_texture != TextureID(-1)) {
            emission *= lod.sample(tm.get_sampler2D(emissive_texture), sd.uv).rgb;
        }

        // occlusion only attenuates ambient/indirect light; not applied in a path tracer.

        const float alpha = ggx_roughness_to_alpha(max(roughness, merian_hint_min_roughness));

        // KHR_materials_anisotropy: stretch alpha along the anisotropy direction and rotate the
        // shading frame so the tangent aligns with it; the bitangent keeps the base roughness.
        float2 specular_alpha = float2(alpha);
        if (merian_hint_enable_anisotropy) {
            float strength = anisotropy_strength;
            float2 dir = float2(cos(anisotropy_rotation), sin(anisotropy_rotation));
            if (anisotropy_texture != TextureID(-1)) {
                const float3 a = lod0.sample(tm.get_sampler2D(anisotropy_texture), sd.uv).rgb;
                const float2 d = normalize(a.xy * 2.0f - 1.0f);
                dir = float2(dir.x * d.x - dir.y * d.y, dir.x * d.y + dir.y * d.x);
                strength *= a.b;
            }
            if (strength > 0.0f) {
                specular_alpha.x = lerp(alpha, 1.0f, strength * strength);
                const float3 t =
                    dir.x * shading_frame.get_tangent() + dir.y * shading_frame.get_bitangent();
                shading_frame = OrthonormalFrame(shading_frame.get_normal(), t);
            }
        }

        // dielectric specular F0 from the IOR (KHR_materials_ior; default 1.5 -> 0.04).
        const float ior_r = (ior - 1.0f) / (ior + 1.0f);
        const float dielectric_f0 = ior_r * ior_r;

        // relative IOR (incident / transmitted); drives the Fresnel split and the refraction.
        const float eta = sd.ior / (sd.front_facing ? ior : 1.0f);
        float tw = 0.0f;
        if (merian_hint_enable_transmission) {
            tw = transmission_weight;
        }

        // iridescence strength / film index / thickness, gated by the toggle.
        float iridescence = 0.0f;
        float iri_ior = 1.0f;
        float iri_thickness = 0.0f;
        if (merian_hint_enable_iridescence) {
            iridescence = iridescence_factor;
            if (iridescence_texture != TextureID(-1)) {
                iridescence *= lod0.sample(tm.get_sampler2D(iridescence_texture), sd.uv).r;
            }
            iri_ior = iridescence_ior;
            // Thickness texture green channel selects between min and max; absent -> max.
            float t = 1.0f;
            if (iridescence_thickness_texture != TextureID(-1)) {
                t = lod0.sample(tm.get_sampler2D(iridescence_thickness_texture), sd.uv).g;
            }
            iri_thickness = lerp(iridescence_thickness_min, iridescence_thickness_max, t);
        }

        // Opaque part: iridescent specular reflection over diffuse. Transmissive part: a coupled
        // rough dielectric whose interface reflectance is iridescent. Mixed by the transmission.
        const GltfOpaqueDielectric opaque_dielectric =
            GltfOpaqueDielectric(SpecularBRDF(specular_alpha), LambertDiffuseBRDF(base.rgb),
                                 dielectric_f0, eta, iri_ior, iri_thickness, iridescence);
        const GltfDielectric dielectric = GltfDielectric(
            opaque_dielectric,
            IridescentRoughDielectricBSDF(alpha, eta, base.rgb, float3(dielectric_f0), iri_ior,
                                          iri_thickness, iridescence),
            tw);
        const GltfReflective reflective =
            GltfReflective(dielectric,
                           GltfConductor(SpecularBRDF(specular_alpha), base.rgb, iri_ior,
                                         iri_thickness, iridescence),
                           metallic);

        // Clearcoat weight and roughness, gated by the toggle.
        float coat_weight = 0.0f;
        float coat_alpha = ggx_roughness_to_alpha(merian_hint_min_roughness);
        if (merian_hint_enable_clearcoat) {
            coat_weight = clearcoat_weight;
            if (clearcoat_texture != TextureID(-1)) {
                coat_weight *= lod0.sample(tm.get_sampler2D(clearcoat_texture), sd.uv).r;
            }
            float coat_rough = clearcoat_roughness;
            if (clearcoat_roughness_texture != TextureID(-1)) {
                coat_rough *= lod0.sample(tm.get_sampler2D(clearcoat_roughness_texture), sd.uv).g;
            }
            coat_alpha = ggx_roughness_to_alpha(max(coat_rough, merian_hint_min_roughness));
        }

        // volume absorption, entered on a front-face transmission.
        float3 absorb = float3(0.0f);
        if (merian_hint_enable_volume && sd.front_facing) {
            absorb = absorption;
        }

        // Sheen colour and roughness, gated by the toggle (colour 0 -> the sheen lobe drops out).
        float3 sheen_col = float3(0.0f);
        float sheen_rough = 0.0f;
        if (merian_hint_enable_sheen) {
            sheen_col = sheen_color;
            if (sheen_color_texture != TextureID(-1)) {
                sheen_col *= lod.sample(tm.get_sampler2D(sheen_color_texture), sd.uv).rgb;
            }
            sheen_rough = sheen_roughness;
            if (sheen_roughness_texture != TextureID(-1)) {
                sheen_rough *= lod0.sample(tm.get_sampler2D(sheen_roughness_texture), sd.uv).a;
            }
        }
        const float max_sheen = max(sheen_col.r, max(sheen_col.g, sheen_col.b));
        const GltfSheened sheened =
            GltfSheened(SheenBRDF(sheen_col, sheen_rough), reflective, sheen_rough, max_sheen);

        GltfMaterialSample s;
        s.brdf =
            GltfBSDF(sheened, GltfCoated(SpecularBRDF(coat_alpha), sheened, GLTF_DIELECTRIC_IOR_F0),
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
        get {
            return shading_frame.get_normal();
        }
    }

    override float get_eta() {
        return eta;
    }

    override float3 get_absorption() {
        return absorption;
    }
}

} // namespace merian
