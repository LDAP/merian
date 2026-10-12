__exported import merian_shaders.shading.shading_function;

namespace merian {

// Directional albedo split by lobe.
struct Albedo {
    float3 diffuse;
    float3 specular;

    __init(const float3 diffuse, const float3 specular) {
        this.diffuse = diffuse;
        this.specular = specular;
    }

    static Albedo diffuse_only(const float3 diffuse) {
        return Albedo(diffuse, float3(0.0f));
    }

    static Albedo specular_only(const float3 specular) {
        return Albedo(float3(0.0f), specular);
    }

    Albedo scaled(const float3 factor) {
        return Albedo(diffuse * factor, specular * factor);
    }

    property float3 reflection {
        get {
            return diffuse + specular;
        }
    }
}

static const float BSDF_MIN_COS_THETA = 1e-6f;

// Below this the microfacet formulation divides by a vanishing half vector and the lobe is
// narrower than a pixel. pbrt-v4, TrowbridgeReitzDistribution::EffectivelySmooth.
static const float BSDF_SMOOTH_ALPHA = 1e-3f;

// Mirror direction in the shading frame.
float3 reflect_z(const float3 wi) {
    return float3(-wi.x, -wi.y, wi.z);
}

interface BSDF : ShadingFunction {

    // The incident direction points away from the
    // shading location.
    Albedo get_albedo(const float3 wi);

    // Pre-integrated specular reflectance for the given F0.
    float3 get_preintegrated_specular(const float3 F0, const float3 wi);

    // GGX alpha (roughness^2) of the whole BSDF; a diffuse lobe counts as roughness 1.
    property float effective_alpha {
        get;
    }

    // GGX alpha (roughness^2) of the specular lobe.
    property float specular_alpha {
        get;
    }
    // The lobes this BSDF can scatter into, as BSDFLobe flags.
    uint get_lobe_types();
}

BSDFLobe
bsdf_lobe_at(const BSDFLobe lobe, const uint lobe_types, const float3 wi, const float3 wo) {
    if (lobe != BSDFLobe::None) {
        return lobe;
    }
    const uint side =
        wi.z * wo.z < 0.0f ? uint(BSDFLobe::Transmission) : uint(BSDFLobe::Reflection);
    return BSDFLobe(lobe_types & side);
}

BSDFLobe pick_lobe(inout RandomGenerator rng,
                   const float selected_pdf0,
                   const float selected_pdf1,
                   const float select1,
                   const BSDFLobe lobe0,
                   const BSDFLobe lobe1) {
    const float u = rng.next_float();
    const float sum = selected_pdf0 + selected_pdf1;
    return (sum > 0.0f ? u * sum < selected_pdf1 : u < select1) ? lobe1 : lobe0;
}

float3 eval_pdf_part<B : BSDF, let LOBE : bool>(const B part,
                                               const float3 wi,
                                               const float3 wo,
                                               inout RandomGenerator rng,
                                               out float pdf,
                                               out BSDFLobe lobe) {
    lobe = BSDFLobe::None;
    if (LOBE) {
        const float3 f = part.eval_pdf_lobe(wi, wo, rng, pdf, lobe);
        lobe = bsdf_lobe_at(lobe, part.get_lobe_types(), wi, wo);
        return f;
    }
    return part.eval_pdf(wi, wo, rng, pdf);
}

struct NullBSDF : BSDF {
    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        return none;
    }
    float pdf(const float3 wi, const float3 wo) {
        return 0;
    }
    float3 eval_pdf(const float3 wi, const float3 wo, inout RandomGenerator rng, out float pdf) {
        pdf = 0;
        return float3(0);
    }
    Albedo get_albedo(const float3 wi) {
        return Albedo(float3(0), float3(0));
    }
    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return float3(0);
    }
    property float effective_alpha {
        get {
            return 1.0f;
        }
    }
    property float specular_alpha {
        get {
            return 1.0f;
        }
    }

    uint get_lobe_types() {
        return uint(BSDFLobe::None);
    }
}

} // namespace merian
