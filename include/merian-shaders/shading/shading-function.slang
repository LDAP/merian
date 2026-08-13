__exported import merian_shaders.utils.random;

namespace merian {

// Scattering smoothness of a sample plus the side of the surface it went to, as independent bits
// so a consumer can test either alone.
[Flags]
enum BSDFLobe {
    Delta = 1,        // singular (perfect mirror / refraction)
    Glossy = 2,       // microfacet
    Diffuse = 4,      // Lambertian / ideal-rough
    Transmission = 8, // scattered through the surface instead of off it
}

// A sample from a BSDF or PhaseFunction.
struct ShadingFunctionSample {
    // the outgoing direction
    float3 wo;
    // probability density in solid angle
    float pdf;
    // sample weight, i.e. f(wi, wo) * dot(wo, n) / pdf = eval() / pdf().
    float3 weight;
    // the lobe this sample was drawn from
    BSDFLobe lobe;

    __init(const float3 wo, const float pdf, const float3 weight) {
        this.wo = wo;
        this.pdf = pdf;
        this.weight = weight;
        this.lobe = BSDFLobe::Glossy;
    }

    __init(const float3 wo, const float pdf, const float3 weight, const BSDFLobe lobe) {
        this.wo = wo;
        this.pdf = pdf;
        this.weight = weight;
        this.lobe = lobe;
    }
}

// Common base for BSDFs and PhaseFunctions.
interface ShadingFunction {

    /** Evaluates the ShadingFunction. BSDFs assume a local tangent space with n = (0, 0, 1), t =
     * (1, 0, 0), b = (0, 1, 0). The incident and outgoing direction point away from the shading
     * location.
     *
     *  Returns solid angle: f(wi, wo) * dot(wo, n)
     */
    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng);

    // -----------------------------

    /** Samples the ShadingFunction. BSDFs assume a local tangent space with n = (0, 0, 1), t = (1,
     * 0, 0), b = (0, 1, 0). The incident and outgoing direction point away from the shading
     * location.
     *
     * Returns an outgoing direction wo.
     */
    Optional<float3> sample(const float3 wi, inout RandomGenerator rng);

    // -----------------------------

    /** Evaluates the pdf of the ShadingFunction. BSDFs assume a local tangent space with n
     * = (0, 0, 1), t = (1, 0, 0), b = (0, 1, 0). The incident and outgoing direction point away
     * from the shading location.
     *
     *  Returns in solid angle.
     */
    float pdf(const float3 wi, const float3 wo);

    /** Samples and evaluates the ShadingFunction. BSDFs assume a local tangent space with n = (0,
     * 0, 1), t = (1, 0, 0), b = (0, 1, 0). The incident and outgoing direction point away from the
     * shading location.
     *
     */
    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng);
}

// Utility for default implementations of sample_eval (Slang issue prevents a default implementation)
Optional<ShadingFunctionSample>
sample_eval_impl<T : ShadingFunction>(const T sf, const float3 wi, inout RandomGenerator rng) {
    const Optional<float3> optional_wo = sf.sample(wi, rng);
    if (let wo = optional_wo) {
        const float p = sf.pdf(wi, wo);
        return ShadingFunctionSample(wo, p, sf.eval(wi, wo, rng) / p);
    }
    return none;
}

}
