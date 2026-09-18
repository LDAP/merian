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

struct NullBSDF : BSDF {
    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        return float3(0);
    }
    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        return none;
    }
    float pdf(const float3 wi, const float3 wo) {
        return 0;
    }
    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        return sample_eval_impl<This>(this, wi, rng);
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
