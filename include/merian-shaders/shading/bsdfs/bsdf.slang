__exported import merian_shaders.shading.shading_function;

namespace merian {

struct Albedo {
    float3 reflection;

    __init(const float3 reflected) {
        reflection = reflected;
    }
}

static const float BSDF_MIN_COS_THETA = 1e-6f;

interface BSDF : ShadingFunction {

    // The incident direction points away from the
    // shading location.
    Albedo get_albedo(const float3 wi);

    // Pre-integrated specular reflectance for the given F0.
    float3 get_preintegrated_specular(const float3 F0, const float3 wi);
}

struct NullBSDF : BSDF {
    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) { return float3(0); }
    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) { return none; }
    float pdf(const float3 wi, const float3 wo) { return 0; }
    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        return sample_eval_impl<This>(this, wi, rng);
    }
    Albedo get_albedo(const float3 wi) { return Albedo(float3(0)); }
    float3 get_preintegrated_specular(const float3 F0, const float3 wi) { return float3(0); }
}

}
