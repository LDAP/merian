#include <gtest/gtest.h>

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <numbers>

using namespace merian;

#ifndef TEST_SHADER_DIR
#define TEST_SHADER_DIR "."
#endif

namespace {

// Analytic integrals over the unit hemisphere (z >= 0), the targets the checks converge to.
constexpr float INTEGRAL_COS_HEMISPHERE = std::numbers::pi_v<float>; // INT cos(theta) dwo
constexpr float PDF_NORMALIZATION = 1.0f;                            // INT pdf dwo

// Monte-Carlo checks pass within this many standard errors of the analytic target
// (~1-in-1e9 false-failure rate for a correct, unbiased estimator).
constexpr float MC_SIGMA = 6.0f;

// Floor on the Monte-Carlo tolerance: a zero-variance estimator (e.g. Lambert, whose
// cos/pdf is constant) reports stderr 0, but its mean still carries the round-off of the
// Welford reduction over 2^18 terms.
constexpr float MC_NUMERIC_FLOOR = 1e-3f;

// Identities that hold exactly in real arithmetic (weight==eval/pdf, pdf round-trips,
// reciprocity) differ only by float round-off accumulated over a handful of operations.
constexpr float FLOAT_EQ_TOL = 1e-4f;

// sample_eval reuses the sampled half-vector for its pdf, while pdf() reconstructs it from
// normalize(wi+wo); the two differ only by round-off, but the NDF amplifies it near the
// specular peak, so the pdf consistency is checked relatively rather than absolutely.
constexpr float PDF_REL_TOL = 1e-3f;

// Midpoint quadrature error of a smooth pdf on the 256x512 grid.
constexpr float QUADRATURE_TOL = 1e-3f;

// Link-time config modules: each exports a concrete-typed make_test_bsdf, so the
// generic bsdf-checks shader is specialized to one BSDF without runtime dispatch.
const char* const CONFIG_LAMBERT = R"(
import merian_shaders.shading.bsdfs.brdf_lambert_diffuse;
import bsdf.bsdf_test_common;
namespace merian_test {
export merian::LambertDiffuseBRDF make_test_bsdf(BSDFParams p) {
    return merian::LambertDiffuseBRDF(p.albedo);
}
}
)";

const char* const CONFIG_GGX = R"(
import merian_shaders.shading.bsdfs.brdf_ggx;
import bsdf.bsdf_test_common;
namespace merian_test {
export merian::GGXBRDF make_test_bsdf(BSDFParams p) {
    return merian::GGXBRDF(p.alpha);
}
}
)";

const char* const CONFIG_GGX_ANISO = R"(
import merian_shaders.shading.bsdfs.brdf_ggx;
import bsdf.bsdf_test_common;
namespace merian_test {
export merian::GGXBRDF make_test_bsdf(BSDFParams p) {
    return merian::GGXBRDF(float2(p.alpha, p.alpha_bitangent));
}
}
)";

const char* const CONFIG_FRESNEL_MIX = R"(
import merian_shaders.shading.bsdfs.bsdf_fresnel_mix;
import merian_shaders.shading.bsdfs.brdf_ggx;
import merian_shaders.shading.bsdfs.brdf_lambert_diffuse;
import bsdf.bsdf_test_common;
namespace merian_test {
export merian::FresnelMixBSDF<merian::GGXBRDF, merian::LambertDiffuseBRDF> make_test_bsdf(BSDFParams p) {
    return merian::FresnelMixBSDF<merian::GGXBRDF, merian::LambertDiffuseBRDF>(
        merian::GGXBRDF(p.alpha), merian::LambertDiffuseBRDF(p.albedo), p.ior_f0);
}
}
)";

const char* const CONFIG_CONDUCTOR = R"(
import merian_shaders.shading.bsdfs.bsdf_conductor_fresnel;
import merian_shaders.shading.bsdfs.brdf_ggx;
import bsdf.bsdf_test_common;
namespace merian_test {
export merian::ConductorFresnelBSDF<merian::GGXBRDF> make_test_bsdf(BSDFParams p) {
    return merian::ConductorFresnelBSDF<merian::GGXBRDF>(merian::GGXBRDF(p.alpha), p.f0);
}
}
)";

} // namespace

class BSDFTest : public ::testing::Test {
  protected:
    static ContextHandle context;
    static ResourceAllocatorHandle allocator;
    static QueueHandle queue;
    static ShaderCompileContextHandle compile_context;

    static void SetUpTestSuite() {
        ContextCreateInfo info{
            .context_extensions = {ExtensionVkValidationLayers::name, ExtensionResources::name},
            .application_name = "test-bsdf",
        };
        context = Context::create(info);
        const auto resources = context->get_context_extension<ExtensionResources>();
        allocator = resources->resource_allocator();
        queue = context->get_queue_GCT();
        compile_context = ShaderCompileContext::create(context);
        compile_context->add_search_path(TEST_SHADER_DIR);
    }

    static void TearDownTestSuite() {
        context->get_device()->get_device().waitIdle();
        compile_context.reset();
        allocator.reset();
        queue.reset();
        context.reset();
    }

    struct BSDFParams {
        float3 albedo{1.0f, 1.0f, 1.0f};
        float alpha = 0.3f;
        float3 f0{1.0f, 1.0f, 1.0f};
        float ior_f0 = 0.04f;
        float alpha_bitangent = 0.3f;
    };

    struct CheckResult {
        float3 furnace;        // average sample_eval weight (directional albedo)
        float3 furnace_stderr; // standard error of furnace
        float mean_cos;        // MC estimate of INT cos(theta) dwo
        float stderr_cos;      // standard error of mean_cos
        uint32_t valid;        // number of accepted samples
        float max_weight_err;  // max |sample_eval.weight - eval/pdf|
        float max_pdf_err;     // max |sample_eval.pdf - pdf(wi,wo)|
        float pdf_integral;    // deterministic quadrature of INT pdf dwo
        float max_recip_err;   // max relative |f(wi,wo) - f(wo,wi)|
    };

    static float as_float(uint32_t u) {
        return std::bit_cast<float>(u);
    }

    CheckResult run(const char* config_src,
                    const float3 wi,
                    const BSDFParams& p,
                    const uint32_t n_samples = (1u << 18),
                    const uint32_t seed = 0x1234567u) {
        const auto composition = SlangComposition::create();
        composition->add_module_from_string("bsdf_config", config_src);
        composition->add_module_from_path("bsdf/bsdf-checks.slang", true);

        const auto program = SlangProgram::create(compile_context, composition);
        const auto entry_point = SlangProgramEntryPoint::create(program, "main");
        const auto pipe_layout = entry_point.get()->get_pipeline_layout(context);
        const auto pipeline = ComputePipeline::create(pipe_layout, entry_point.get()->specialize());
        const auto obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);

        constexpr uint32_t SLOTS = 13;
        const auto output_buffer = allocator->create_buffer(
            SLOTS * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::HOST_ACCESS_RANDOM, "bsdf_output");

        const auto params = entry_point.get()->create_shader_object(context, "params", allocator);
        auto cursor = params->get_cursor();
        cursor["wi"] = wi;
        cursor["n_samples"] = n_samples;
        cursor["seed"] = seed;
        cursor["p"]["albedo"] = p.albedo;
        cursor["p"]["alpha"] = p.alpha;
        cursor["p"]["f0"] = p.f0;
        cursor["p"]["ior_f0"] = p.ior_f0;
        cursor["p"]["alpha_bitangent"] = p.alpha_bitangent;
        cursor["output"] = output_buffer;

        queue->submit_wait([&](const CommandBufferHandle& cmd) {
            cmd->bind(pipeline);
            entry_point.get()->bind_entry_point_parameter("params", params, cmd, pipeline,
                                                          obj_allocator);
            cmd->dispatch(1, 1, 1);
        });

        uint32_t slots[SLOTS];
        const auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
        std::memcpy(slots, mapped, sizeof(slots));
        output_buffer->get_memory()->unmap();

        const CheckResult r{{as_float(slots[0]), as_float(slots[1]), as_float(slots[2])},
                            {as_float(slots[3]), as_float(slots[4]), as_float(slots[5])},
                            as_float(slots[6]),
                            as_float(slots[7]),
                            slots[8],
                            as_float(slots[9]),
                            as_float(slots[10]),
                            as_float(slots[11]),
                            as_float(slots[12])};
        std::fprintf(stderr,
                     "DUMP furnace=%.6g,%.6g,%.6g fse=%.3g,%.3g,%.3g meancos=%.7f secos=%.4g "
                     "werr=%.4g perr=%.4g pdfint=%.7f reciperr=%.4g acc=%.5f\n",
                     r.furnace.x, r.furnace.y, r.furnace.z, r.furnace_stderr.x, r.furnace_stderr.y,
                     r.furnace_stderr.z, r.mean_cos, r.stderr_cos, r.max_weight_err, r.max_pdf_err,
                     r.pdf_integral, r.max_recip_err, float(r.valid) / float(n_samples));
        return r;
    }

    // sample() must draw from exactly the distribution pdf() reports: the importance
    // sampled estimate of INT cos(theta) dwo then converges to its analytic value.
    void expect_sample_pdf_match(const CheckResult& r) {
        ASSERT_GT(r.valid, 0u);
        const float tol = MC_SIGMA * r.stderr_cos + MC_NUMERIC_FLOOR;
        EXPECT_NEAR(r.mean_cos, INTEGRAL_COS_HEMISPHERE, tol) << "INT cos dwo";
    }

    // sample_eval's reported pdf and weight are equal to the standalone pdf()/eval()
    // definitions up to float round-off.
    void expect_sample_eval_consistent(const CheckResult& r) {
        EXPECT_LT(r.max_pdf_err, PDF_REL_TOL) << "sample_eval.pdf vs pdf() (relative)";
        EXPECT_LT(r.max_weight_err, FLOAT_EQ_TOL) << "sample_eval.weight vs eval/pdf";
    }

    // Helmholtz reciprocity f(wi,wo) == f(wo,wi); the two eval() paths differ only by
    // float round-off.
    void expect_reciprocal(const CheckResult& r) {
        EXPECT_LT(r.max_recip_err, FLOAT_EQ_TOL) << "reciprocity";
    }

    // Directional albedo is finite and gains no energy. Only valid for energy-conserving
    // BSDFs; the Fresnel-layered heuristic is excluded (it lacks a multiscatter term).
    void expect_energy_conserving(const CheckResult& r) {
        EXPECT_TRUE(std::isfinite(r.furnace.x) && std::isfinite(r.furnace.y) &&
                    std::isfinite(r.furnace.z));
        EXPECT_LE(r.furnace.x, 1.0f + MC_SIGMA * r.furnace_stderr.x) << "furnace r";
        EXPECT_LE(r.furnace.y, 1.0f + MC_SIGMA * r.furnace_stderr.y) << "furnace g";
        EXPECT_LE(r.furnace.z, 1.0f + MC_SIGMA * r.furnace_stderr.z) << "furnace b";
    }

    // pdf integrates to 1 over the hemisphere. Only valid for samplers that never place
    // wo below the horizon (otherwise the mass equals the acceptance rate < 1).
    void expect_pdf_normalized(const CheckResult& r) {
        EXPECT_NEAR(r.pdf_integral, PDF_NORMALIZATION, QUADRATURE_TOL) << "INT pdf dwo";
    }
};

ContextHandle BSDFTest::context;
ResourceAllocatorHandle BSDFTest::allocator;
QueueHandle BSDFTest::queue;
ShaderCompileContextHandle BSDFTest::compile_context;

float3 wi_from_zenith(const float degrees) {
    const float theta = degrees * (std::numbers::pi_v<float> / 180.0f);
    return {std::sin(theta), 0.0f, std::cos(theta)};
}

// Normal-ish and grazing incidence (the latter stresses the VNDF reflection Jacobian).
static const float3 WI_NORMAL = wi_from_zenith(0.0f);
static const float3 WI_GRAZING = wi_from_zenith(80.0f);

// Lambert
TEST_F(BSDFTest, LambertNormal) {
    BSDFParams p;
    p.albedo = {0.8f, 0.5f, 0.2f};
    const auto r = run(CONFIG_LAMBERT, WI_NORMAL, p);
    expect_sample_pdf_match(r);
    expect_sample_eval_consistent(r);
    expect_reciprocal(r);
    expect_energy_conserving(r);
    expect_pdf_normalized(r);
    // Lambert furnace == albedo.
    EXPECT_NEAR(r.furnace.x, p.albedo.x, MC_SIGMA * r.furnace_stderr.x);
    EXPECT_NEAR(r.furnace.y, p.albedo.y, MC_SIGMA * r.furnace_stderr.y);
    EXPECT_NEAR(r.furnace.z, p.albedo.z, MC_SIGMA * r.furnace_stderr.z);
}

TEST_F(BSDFTest, LambertGrazing) {
    BSDFParams p;
    p.albedo = {0.8f, 0.5f, 0.2f};
    const auto r = run(CONFIG_LAMBERT, WI_GRAZING, p);
    expect_sample_pdf_match(r);
    expect_sample_eval_consistent(r);
    expect_reciprocal(r);
    expect_energy_conserving(r);
    expect_pdf_normalized(r);
    EXPECT_NEAR(r.furnace.x, p.albedo.x, MC_SIGMA * r.furnace_stderr.x);
}

// GGX (varying roughness). pdf normalization is left out: VNDF reflection sends part of
// the lobe below the horizon, so INT pdf dwo equals the acceptance rate, not 1.
class GGXRoughness : public BSDFTest, public ::testing::WithParamInterface<float> {
  protected:
    void check(const float3 wi) {
        BSDFParams p;
        p.alpha = GetParam();
        const auto r = run(CONFIG_GGX, wi, p);
        expect_sample_pdf_match(r);
        expect_sample_eval_consistent(r);
        expect_reciprocal(r);
        expect_energy_conserving(r);
    }
};

TEST_P(GGXRoughness, Normal) {
    check(WI_NORMAL);
}
TEST_P(GGXRoughness, Grazing) {
    check(WI_GRAZING);
}

INSTANTIATE_TEST_SUITE_P(Alpha, GGXRoughness, ::testing::Values(0.05f, 0.3f, 0.8f));

// Anisotropic GGX. wi is placed off the alpha axes so the cross terms of D, G and the
// VNDF warp are exercised; includes the extreme tangent stretch of a full-strength
// KHR_materials_anisotropy material at minimal base roughness.
class GGXAnisotropy : public BSDFTest,
                      public ::testing::WithParamInterface<std::pair<float, float>> {
  protected:
    void check(const float3 wi) {
        BSDFParams p;
        p.alpha = GetParam().first;
        p.alpha_bitangent = GetParam().second;
        const auto r = run(CONFIG_GGX_ANISO, wi, p);
        expect_sample_pdf_match(r);
        expect_sample_eval_consistent(r);
        expect_reciprocal(r);
        expect_energy_conserving(r);
    }
};

float3 wi_from_zenith_azimuth(const float zenith_deg, const float azimuth_deg) {
    constexpr float TO_RAD = std::numbers::pi_v<float> / 180.0f;
    const float theta = zenith_deg * TO_RAD;
    const float phi = azimuth_deg * TO_RAD;
    return {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
}

TEST_P(GGXAnisotropy, Normal) {
    check(wi_from_zenith_azimuth(30.0f, 40.0f));
}
TEST_P(GGXAnisotropy, Grazing) {
    check(wi_from_zenith_azimuth(80.0f, 40.0f));
}

// The 100:1 stretch is the practical extreme; beyond it the cos/pdf estimator of the
// sampler check is too heavy-tailed to converge (the lobe degenerates to a 1D sheet).
INSTANTIATE_TEST_SUITE_P(Alpha,
                         GGXAnisotropy,
                         ::testing::Values(std::pair{0.5f, 0.05f},
                                           std::pair{0.05f, 0.5f},
                                           std::pair{1.0f, 0.01f},
                                           std::pair{0.3f, 0.1f}));

// FresnelMix<GGX, Lambert>. Energy conservation is left out: the layered model is a
// heuristic (no multiscatter term) and slightly gains energy at grazing.
class FresnelMixRoughness : public BSDFTest, public ::testing::WithParamInterface<float> {
  protected:
    void check(const float3 wi) {
        BSDFParams p;
        p.alpha = GetParam();
        p.albedo = {0.7f, 0.7f, 0.7f};
        const auto r = run(CONFIG_FRESNEL_MIX, wi, p);
        expect_sample_pdf_match(r);
        expect_sample_eval_consistent(r);
        expect_reciprocal(r);
    }
};

TEST_P(FresnelMixRoughness, Normal) {
    check(WI_NORMAL);
}
TEST_P(FresnelMixRoughness, Grazing) {
    check(WI_GRAZING);
}

INSTANTIATE_TEST_SUITE_P(Alpha, FresnelMixRoughness, ::testing::Values(0.05f, 0.3f, 0.8f));

// ConductorFresnel<GGX>. Delegates sampling to GGX and scales eval by Fresnel (<= 1), so
// it stays energy-conserving; pdf normalization is left out for the same reason as GGX.
class ConductorRoughness : public BSDFTest, public ::testing::WithParamInterface<float> {
  protected:
    void check(const float3 wi) {
        BSDFParams p;
        p.alpha = GetParam();
        p.f0 = {0.95f, 0.64f, 0.54f}; // copper-ish
        const auto r = run(CONFIG_CONDUCTOR, wi, p);
        expect_sample_pdf_match(r);
        expect_sample_eval_consistent(r);
        expect_reciprocal(r);
        expect_energy_conserving(r);
    }
};

TEST_P(ConductorRoughness, Normal) {
    check(WI_NORMAL);
}
TEST_P(ConductorRoughness, Grazing) {
    check(WI_GRAZING);
}

INSTANTIATE_TEST_SUITE_P(Alpha, ConductorRoughness, ::testing::Values(0.05f, 0.3f, 0.8f));
