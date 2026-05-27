import bsdf;

namespace merian {

struct MixBSDF<B0 : BSDF, B1 : BSDF> : BSDF {
    B0 bsdf0;
    B1 bsdf1;
    float weight;

    __init(B0 bsdf0, B1 bsdf1, float weight) {
        this.bsdf0 = bsdf0;
        this.bsdf1 = bsdf1;
        this.weight = weight;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        return lerp(bsdf0.eval(wi, wo, rng), bsdf1.eval(wi, wo, rng), weight);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        if (rng.next_float() < weight) {
            return bsdf1.sample(wi, rng);
        }
        return bsdf0.sample(wi, rng);
    }

    float pdf(const float3 wi, const float3 wo) {
        return lerp(bsdf0.pdf(wi, wo), bsdf1.pdf(wi, wo), weight);
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(
            lerp(bsdf0.get_albedo(wi).reflection, bsdf1.get_albedo(wi).reflection, weight));
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return lerp(bsdf0.get_preintegrated_specular(F0, wi),
                    bsdf1.get_preintegrated_specular(F0, wi), weight);
    }
}

}
