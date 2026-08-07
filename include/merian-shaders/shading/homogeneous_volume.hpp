#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/utils/properties.hpp"
#include "merian/utils/vector_matrix.hpp"

#include <cmath>
#include <memory>
#include <string>

namespace merian {

// The medium filling a scene outside of any surface. The concrete type is selected at composition
// time (see Scene::set_exterior_volume), so Vacuum removes the medium from the renderers entirely
// instead of being branched around.
class HomogeneousVolume {
  public:
    virtual ~HomogeneousVolume() = default;

    virtual SlangComposition::SlangModule get_slang_module() const = 0;

    virtual std::string get_type_name() const = 0;

    virtual void write_to(ShaderCursor cursor) const = 0;

    virtual void properties([[maybe_unused]] Properties& props) {}
};

using HomogeneousVolumeHandle = std::shared_ptr<HomogeneousVolume>;

class VacuumVolume : public HomogeneousVolume {
  public:
    SlangComposition::SlangModule get_slang_module() const override {
        return SlangComposition::SlangModule::from_path(
            "merian-shaders/shading/homogeneous-volume.slang", false);
    }

    std::string get_type_name() const override {
        return "merian::Vacuum";
    }

    void write_to([[maybe_unused]] ShaderCursor cursor) const override {}
};

// The HG + Draine mixture parameters for a water droplet diameter in micrometer.
// Jendersie and d'Eon, "An Approximate Mie Scattering Function for Fog and Cloud Rendering",
// SIGGRAPH 2023 Talks; eq. 4-7 plus the small-particle mappings from the supplemental.
struct MieApproxFit {
    float hg_g;
    float d_g;
    float d_a;
    float w_d;

    static MieApproxFit for_diameter(const float d) {
        const float l = std::log(d);
        if (d <= 0.1f) {
            return {13.8f * d * d, 1.1456f * d * std::sin(9.29044f * d), 250.f,
                    0.252977f - 312.983f * std::pow(d, 4.3f)};
        }
        if (d < 1.5f) {
            return {0.862f - 0.143f * l * l,
                    0.379685f * std::cos(1.19692f * std::cos(((l - 0.238604f) * (l + 1.00667f)) /
                                                             (0.507522f - 0.15677f * l)) +
                                         1.37932f * l + 0.0625835f) +
                        0.344213f,
                    250.f,
                    0.146209f * std::cos(3.38707f * l + 2.11193f) + 0.316072f + 0.0778917f * l};
        }
        if (d < 5.f) {
            return {0.0604931f * std::log(l) + 0.940256f,
                    0.500411f - 0.081287f / (-2.f * l + std::tan(l) + 1.27551f),
                    7.30354f * l + 6.31675f,
                    0.026914f * (l - std::cos(5.68947f * (std::log(l) - 0.0292149f))) + 0.376475f};
        }
        return {std::exp(-0.0990567f / (d - 1.67154f)),
                std::exp(-2.20679f / (d + 3.91029f) - 0.428934f),
                std::exp(3.62489f - 8.29288f / (d + 5.52825f)),
                std::exp(-0.599085f / (d - 0.641583f) - 0.665888f)};
    }
};

class FogVolume : public HomogeneousVolume {
  public:
    SlangComposition::SlangModule get_slang_module() const override {
        return SlangComposition::SlangModule::from_path(
            "merian-shaders/shading/homogeneous-volume.slang", false);
    }

    std::string get_type_name() const override {
        return "merian::Fog";
    }

    void write_to(ShaderCursor cursor) const override {
        cursor["mu_t"] = mu_t;
        cursor["mu_s"] = mu_s;
        cursor["max_distance"] = max_distance;

        const MieApproxFit fit = MieApproxFit::for_diameter(particle_size_um);
        auto phase = cursor["phase"];
        phase["hg_g"] = fit.hg_g;
        phase["d_g"] = fit.d_g;
        phase["d_a"] = fit.d_a;
        phase["w_d"] = fit.w_d;
    }

    void properties(Properties& props) override {
        props.config_float("particle size", particle_size_um,
                           "Water droplet diameter in micrometer. Fog is roughly 5 - 15, cloud "
                           "droplets reach 50; below 0.1 the phase function tends to Rayleigh.",
                           0.1f, 0.001f, 50.f);
        props.config_float("max distance", max_distance,
                           "Distance at which the optical depth saturates.");
    }

    float3 mu_t{0.f};
    float3 mu_s{0.f};
    float max_distance = 1e7f;
    float particle_size_um = 7.f;
};

} // namespace merian
