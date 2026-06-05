// Generic BSDF correctness checks. The concrete BSDF is bound at link time:
// the imported `bsdf_config` module exports merian_test::make_test_bsdf, which
// is regenerated per-type on the host.
//
// One dispatch fills the output buffer; slot layout mirrors CheckResult on the
// host side. Monte-Carlo estimators report their standard error so the host can
// set statistical tolerances; deterministic grid checks report a max error.

#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.shading.bsdfs.bsdf;
import merian_shaders.utils.random;
import bsdf.bsdf_test_common;
import bsdf_config;

struct Params {
    float3 wi;
    uint n_samples;
    uint seed;
    merian_test::BSDFParams p;
    RWStructuredBuffer<uint> output;
};

// Folds a non-finite value into a sticky sentinel, since max(acc, NaN) would
// otherwise keep acc and silently hide a NaN/Inf from the error checks.
float accumulate_err(const float acc, const float v) {
    return isfinite(v) ? max(acc, v) : 1e30f;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    let bsdf = merian_test::make_test_bsdf(params.p);
    const float3 wi = params.wi;

    merian::RandomGenerator rng = merian::XorShift32(params.seed | 1u);

    // 1. Monte-Carlo loop: furnace (directional albedo), sample<->pdf agreement, weight.
    // Means and variances use Welford's online update: the accumulators stay at the
    // estimate's magnitude (so the reduction does not lose precision over 2^N terms) and
    // the variance avoids the catastrophic cancellation of E[x^2] - E[x]^2.
    float3 mean_weight = float3(0);
    float3 m2_weight = float3(0);
    float mean_cos = 0; // estimator of INT cos(theta) dwo
    float m2_cos = 0;
    uint valid = 0;
    float max_weight_err = 0;
    float max_pdf_err = 0;

    for (uint i = 0; i < params.n_samples; ++i) {
        // A below-horizon / clamped draw contributes zero and is still counted
        // in the N average, so the estimators stay unbiased.
        float3 weight = float3(0);
        float c = 0;
        if (let s = bsdf.sample_eval(wi, rng)) {
            valid++;
            weight = s.weight;
            c = s.wo.z / s.pdf;

            // weight and pdf must match the independent eval()/pdf() definition. pdf() rebuilds
            // the half-vector via normalize(wi + wo) while sample_eval may reuse the sampled h;
            // the NDF amplifies that round-off near the specular peak, so the pdf is compared in
            // relative terms.
            const float pdf_ref = bsdf.pdf(wi, s.wo);
            const float3 eval_ref = bsdf.eval(wi, s.wo, rng);
            max_pdf_err = accumulate_err(
                max_pdf_err, abs(s.pdf - pdf_ref) / (max(abs(s.pdf), abs(pdf_ref)) + 1e-6f));
            if (pdf_ref > 0) {
                max_weight_err =
                    accumulate_err(max_weight_err, length(s.weight - (eval_ref / pdf_ref)));
            }
        }

        const float inv_i = 1.0f / float(i + 1);
        const float3 delta_weight = weight - mean_weight;
        mean_weight += delta_weight * inv_i;
        m2_weight += delta_weight * (weight - mean_weight);
        const float delta_cos = c - mean_cos;
        mean_cos += delta_cos * inv_i;
        m2_cos += delta_cos * (c - mean_cos);
    }

    // stderr of the mean = sqrt(population_variance / N) = sqrt(M2) / N.
    const float inv_n = 1.0f / float(params.n_samples);
    const float3 furnace = mean_weight;
    const float3 furnace_stderr = sqrt(m2_weight) * inv_n;
    const float stderr_cos = sqrt(m2_cos) * inv_n;

    // 2. Deterministic hemisphere quadrature: pdf normalization, reciprocity
    const uint NT = 256;
    const uint NP = 512;
    const float dtheta = (MERIAN_PI * 0.5f) / float(NT);
    const float dphi = MERIAN_TWO_PI / float(NP);
    float pdf_integral = 0;
    float max_recip_err = 0;

    for (uint ti = 0; ti < NT; ++ti) {
        const float theta = (float(ti) + 0.5f) * dtheta;
        const float ct = cos(theta);
        const float st = sin(theta);
        for (uint pi = 0; pi < NP; ++pi) {
            const float phi = (float(pi) + 0.5f) * dphi;
            const float3 wo = float3(st * cos(phi), st * sin(phi), ct);

            pdf_integral += bsdf.pdf(wi, wo) * st * dtheta * dphi;

            // eval() carries the cos(theta_o) factor, so divide it out to compare the BRDF.
            const float3 f_io = bsdf.eval(wi, wo, rng) / wo.z;
            const float3 f_oi = bsdf.eval(wo, wi, rng) / wi.z;
            const float3 d = abs(f_io - f_oi);
            const float3 m = max(abs(f_io), abs(f_oi)) + 1e-6f;
            max_recip_err = accumulate_err(max_recip_err, max(max(d.x / m.x, d.y / m.y), d.z / m.z));
        }
    }

    params.output[0] = asuint(furnace.r);
    params.output[1] = asuint(furnace.g);
    params.output[2] = asuint(furnace.b);
    params.output[3] = asuint(furnace_stderr.r);
    params.output[4] = asuint(furnace_stderr.g);
    params.output[5] = asuint(furnace_stderr.b);
    params.output[6] = asuint(mean_cos);
    params.output[7] = asuint(stderr_cos);
    params.output[8] = valid;
    params.output[9] = asuint(max_weight_err);
    params.output[10] = asuint(max_pdf_err);
    params.output[11] = asuint(pdf_integral);
    params.output[12] = asuint(max_recip_err);
}
