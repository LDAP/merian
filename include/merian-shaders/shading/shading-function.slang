__exported import merian_shaders.utils.random;

namespace merian {

// Reflection and transmission occupy separate nibbles.
enum BSDFLobe : uint {
    None = 0x00,

    DiffuseReflection = 0x01,  // Lambertian / ideal-rough
    GlossyReflection = 0x02,   // microfacet
    DeltaReflection = 0x04,    // singular (perfect mirror)

    DiffuseTransmission = 0x10,
    GlossyTransmission = 0x20,
    DeltaTransmission = 0x40, // singular (perfect refraction)

    Reflection = 0x07,
    Transmission = 0x70,
    Delta = 0x44,
    NonDelta = 0x33,
    All = 0x77,
}

// The BSDF can scatter to the far side of the surface.
bool bsdf_is_transmissive(const uint lobes) {
    return (lobes & uint(BSDFLobe::Transmission)) != 0;
}

// Every lobe is singular, so no finite-density sampling strategy can hit this BSDF.
bool bsdf_is_delta_only(const uint lobes) {
    return lobes != 0 && (lobes & uint(BSDFLobe::NonDelta)) == 0;
}

// The sample came from a singular lobe, whichever side it scattered to.
bool bsdf_lobe_is_delta(const BSDFLobe lobe) {
    return (uint(lobe) & uint(BSDFLobe::Delta)) != 0;
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
        this.lobe = BSDFLobe::GlossyReflection;
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

// Utility for default implementations of sample_eval (Slang issue prevents a default
// implementation)
Optional<ShadingFunctionSample>
sample_eval_impl<T : ShadingFunction>(const T sf, const float3 wi, inout RandomGenerator rng) {
    const Optional<float3> optional_wo = sf.sample(wi, rng);
    if (let wo = optional_wo) {
        const float p = sf.pdf(wi, wo);
        return ShadingFunctionSample(wo, p, sf.eval(wi, wo, rng) / p);
    }
    return none;
}

} // namespace merian
