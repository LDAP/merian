__exported import merian_shaders.sampling.guiding;

namespace merian {

enum NEEMode {
    Off,
    Mixture,
    Resampled,
}

enum ScatterMode {
    Off,
    MIS,
    RIS,
}

enum GuidingDirectTarget {
    Full,
    MIS,
    None,
}

enum DebugOutput {
    ScatterStatistics,
    Guiding,
    NEE,
    DistanceGuiding,
    None,
}

extern static const DebugOutput merian_render_debug_output = DebugOutput::None;
extern static const ScatterMode merian_render_scatter_mode = ScatterMode::MIS;
extern static const int merian_render_scatter_candidates = 2;
extern static const bool merian_render_guiding_scale_with_alpha = true;
extern static const float merian_render_guiding_alpha_threshold = 0.05;
extern static const GuidingDirectTarget merian_render_guiding_direct_target =
    GuidingDirectTarget::Full;

float guided_share<G : GuidingDistribution<float3>>(const G guiding,
                                                    const float share,
                                                    const float alpha) {
    if (!guiding.is_valid() || alpha < merian_render_guiding_alpha_threshold) {
        return 0.f;
    }
    const float scaled = merian_render_guiding_scale_with_alpha ? share * alpha : share;
    return clamp(scaled * guiding.get_confidence(), 0.f, 1.f);
}

float guiding_direct_scale(const float scatter_share) {
    switch (merian_render_guiding_direct_target) {
    case GuidingDirectTarget::MIS:
        return scatter_share;
    case GuidingDirectTarget::None:
        return 0.f;
    default:
        return 1.f;
    }
}

Optional<ScatterSample>
sample_scatter<B : ShadingFunction, G : GuidingDistribution<float3>>(const ScatterMixture mixture,
                                                                     const B bsdf,
                                                                     const G guiding,
                                                                     const OrthonormalFrame frame,
                                                                     const float3 wi_local,
                                                                     inout RandomGenerator rng) {
    return merian_render_scatter_mode == ScatterMode::RIS
               ? mixture.sample_ris(bsdf, guiding, frame, merian_render_scatter_candidates,
                                    wi_local, rng)
               : mixture.sample(bsdf, guiding, frame, wi_local, rng);
}

float balance_heuristic(const float pdf, const float other_pdf) {
    return pdf / (pdf + other_pdf);
}

// One-sample MIS (Veach 1997, Sec. 9.2.4)
struct LightOrScatter {
    float p_light;

    __init(const float p_light) {
        this.p_light = p_light;
    }

    bool draws_light(inout RandomGenerator rng) {
        return p_light > 0.f && rng.next_float() < p_light;
    }

    float scatter_density(const float scatter_pdf) {
        return (1.f - p_light) * scatter_pdf;
    }

    float light_density(const float light_pdf) {
        return p_light * light_pdf;
    }
}

} // namespace merian
