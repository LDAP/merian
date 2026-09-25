__exported import merian_shaders.shading.bsdfs.bsdf;
__exported import merian_shaders.utils.random;

namespace merian {

// The identities a BSDF must satisfy, measured one draw and one direction at a time.

// max(acc, NaN) keeps acc, so a non-finite value folds into this sentinel instead.
static const float BSDF_VALIDATE_SENTINEL = 1e30f;

// Round-off in an identity that holds exactly in real arithmetic; a sharp NDF amplifies it
// near the specular peak.
static const float BSDF_VALIDATE_IDENTITY_TOL = 1e-3f;

float bsdf_validate_err(const float acc, const float value) {
    return isfinite(value) ? max(acc, value) : BSDF_VALIDATE_SENTINEL;
}

struct BSDFSampleCheck {
    float3 wo;
    // f * cos / pdf, zero for a rejected draw so the mean over all draws stays unbiased:
    // E[weight] is the directional albedo, which energy conservation bounds by 1
    float3 weight;
    // wo.z / pdf, whose mean estimates INT cos(theta) dwo over the sampled support
    float cos_over_pdf;
    // relative deviation of the sample's own weight and pdf from eval() and pdf(), 0 for a
    // singular draw
    float err_weight;
    float err_pdf;
    // eval() and pdf() must vanish at a singular direction: the largest of them
    float err_delta;
    BSDFLobe lobe;
    bool valid;       // sample() returned a direction
    bool delta;       // the draw came from a singular lobe
    bool transmitted; // the direction crossed the surface
    bool sane;        // finite, non-negative, unit length
};

// One draw from sample_eval(), measured against the free-standing eval() and pdf().
BSDFSampleCheck
bsdf_validate_sample<B : BSDF>(const B bsdf, const float3 wi, inout RandomGenerator rng) {
    BSDFSampleCheck check;
    check.wo = float3(0.f);
    check.weight = float3(0.f);
    check.cos_over_pdf = 0.f;
    check.err_weight = 0.f;
    check.err_pdf = 0.f;
    check.err_delta = 0.f;
    check.lobe = BSDFLobe::None;
    check.valid = false;
    check.delta = false;
    check.transmitted = false;
    check.sane = true;
    if (let smp = bsdf.sample_eval(wi, rng)) {
        check.valid = true;
        check.wo = smp.wo;
        check.weight = smp.weight;
        check.cos_over_pdf = smp.wo.z / smp.pdf;
        check.lobe = smp.lobe;
        check.delta = bsdf_lobe_is_delta(smp.lobe);
        check.transmitted = smp.wo.z * wi.z < 0.f;

        const float3 eval = bsdf.eval(wi, smp.wo, rng);
        const float pdf = bsdf.pdf(wi, smp.wo);
        if (check.delta) {
            check.err_delta = bsdf_validate_err(0.f, pdf);
            check.err_delta = bsdf_validate_err(check.err_delta, max(max(eval.x, eval.y), eval.z));
            check.err_delta = bsdf_validate_err(
                check.err_delta, all(isfinite(smp.weight)) ? 0.f : BSDF_VALIDATE_SENTINEL);
        } else {
            // pdf() rebuilds the half vector from wi + wo while sample_eval may reuse the
            // sampled one, so compare relatively
            check.err_pdf =
                bsdf_validate_err(0.f, abs(smp.pdf - pdf) / (max(abs(smp.pdf), abs(pdf)) + 1e-6f));
            if (pdf > 0.f) {
                const float3 reference = eval / pdf;
                check.err_weight = bsdf_validate_err(0.f, length(smp.weight - reference) /
                                                              (length(reference) + 1e-6f));
            }
        }
        check.sane = all(isfinite(eval)) && all(eval >= 0.f) && isfinite(pdf) && pdf >= 0.f &&
                     isfinite(smp.pdf) && smp.pdf > 0.f && all(isfinite(smp.weight)) &&
                     abs(length(smp.wo) - 1.f) < 1e-3f;
    }
    return check;
}

struct BSDFDirectionCheck {
    float pdf;   // integrates to 1 over the sphere
    float3 eval; // f * cos, integrates to the directional albedo
    // |f(wi, wo) - f(wo, wi)| relative, and 0 where symmetry does not apply: refraction scales
    // radiance by eta^2 between the two sides, so only reflection is reciprocal
    float err_reciprocity;
    // pdf and eval were finite; a non-finite direction contributes zero to the integrals
    bool sane;
};

// One direction of a quadrature over the sphere: the integrals and the symmetry of the BSDF.
BSDFDirectionCheck bsdf_validate_direction<B : BSDF>(const B bsdf,
                                                     const float3 wi,
                                                     const float3 wo,
                                                     inout RandomGenerator rng) {
    BSDFDirectionCheck check;
    const float pdf = bsdf.pdf(wi, wo);
    const float3 eval = bsdf.eval(wi, wo, rng);
    check.sane = isfinite(pdf) && all(isfinite(eval));
    check.pdf = isfinite(pdf) ? pdf : 0.f;
    check.eval = all(isfinite(eval)) ? eval : float3(0.f);
    check.err_reciprocity = 0.f;
    if (wo.z * wi.z > 0.f) {
        // eval() carries the cosine, so divide it out to compare the BSDF itself
        const float3 f_io = eval / wo.z;
        const float3 f_oi = bsdf.eval(wo, wi, rng) / wi.z;
        const float3 difference = abs(f_io - f_oi);
        const float3 scale = max(abs(f_io), abs(f_oi)) + 1e-6f;
        check.err_reciprocity = bsdf_validate_err(
            0.f, max(max(difference.x / scale.x, difference.y / scale.y), difference.z / scale.z));
    }
    return check;
}

} // namespace merian
