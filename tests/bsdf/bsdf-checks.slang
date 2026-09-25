// Generic BSDF correctness checks. The concrete BSDF is bound at link time:
// the imported `bsdf_config` module exports merian_test::make_test_bsdf, which
// is regenerated per-type on the host.
//
// One dispatch fills the output buffer; slot layout mirrors CheckResult on the
// host side. Monte-Carlo estimators report their standard error so the host can
// set statistical tolerances; deterministic grid checks report a max error.

#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.shading.bsdfs.bsdf;
import merian_shaders.shading.bsdfs.bsdf_validate;
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

[shader("compute")][numthreads(1, 1, 1)] void main(ParameterBlock<Params> params) {
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
    uint delta_samples = 0;
    float max_weight_err = 0;
    float max_pdf_err = 0;
    float max_delta_err = 0;

    for (uint i = 0; i < params.n_samples; ++i) {
        // A below-horizon / clamped draw contributes zero and is still counted
        // in the N average, so the estimators stay unbiased.
        const merian::BSDFSampleCheck check = merian::bsdf_validate_sample(bsdf, wi, rng);
        valid += check.valid ? 1 : 0;
        delta_samples += check.delta ? 1 : 0;
        max_delta_err = merian::bsdf_validate_err(max_delta_err, check.err_delta);
        max_pdf_err = merian::bsdf_validate_err(max_pdf_err, check.err_pdf);
        max_weight_err = merian::bsdf_validate_err(max_weight_err, check.err_weight);

        // the albedo is the energy fraction; transmission also compresses radiance by eta^2
        const float3 weight =
            check.transmitted ? check.weight / (params.p.eta * params.p.eta) : check.weight;

        const float inv_i = 1.0f / float(i + 1);
        const float3 delta_weight = weight - mean_weight;
        mean_weight += delta_weight * inv_i;
        m2_weight += delta_weight * (weight - mean_weight);
        const float delta_cos = check.cos_over_pdf - mean_cos;
        mean_cos += delta_cos * inv_i;
        m2_cos += delta_cos * (check.cos_over_pdf - mean_cos);
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
    bool quadrature_sane = true;

    for (uint ti = 0; ti < NT; ++ti) {
        const float theta = (float(ti) + 0.5f) * dtheta;
        const float ct = cos(theta);
        const float st = sin(theta);
        for (uint pi = 0; pi < NP; ++pi) {
            const float phi = (float(pi) + 0.5f) * dphi;
            const float3 wo = float3(st * cos(phi), st * sin(phi), ct);

            const merian::BSDFDirectionCheck check =
                merian::bsdf_validate_direction(bsdf, wi, wo, rng);
            pdf_integral += check.pdf * st * dtheta * dphi;
            max_recip_err = merian::bsdf_validate_err(max_recip_err, check.err_reciprocity);
            quadrature_sane = quadrature_sane && check.sane;
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
    // get_albedo must equal the directional albedo (the furnace value): it is the BSDF throughput
    // a denoiser divides out, so the maximal divisible (high-frequency) content belongs in it.
    const float3 albedo = bsdf.get_albedo(wi).reflection;

    params.output[11] = asuint(quadrature_sane ? pdf_integral : merian::BSDF_VALIDATE_SENTINEL);
    params.output[12] = asuint(max_recip_err);
    params.output[13] = asuint(albedo.r);
    params.output[14] = asuint(albedo.g);
    params.output[15] = asuint(albedo.b);
    params.output[16] = delta_samples;
    params.output[17] = asuint(max_delta_err);
}
