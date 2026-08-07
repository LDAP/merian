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

// Scattering medium with the Mie phase function of Jendersie and d'Eon (2023). The four fit
// coefficients are evaluated here so the shader does not repeat them per sample.
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
        // fit is valid for droplet diameters of 5 - 50 um
        const float d = std::max(particle_size_um, 1.7f);
        auto phase = cursor["phase"];
        phase["hg_g"] = std::exp(-0.0990567f / (d - 1.67154f));
        phase["d_g"] = std::exp(-2.20679f / (d + 3.91029f) - 0.428934f);
        phase["d_a"] = std::exp(3.62489f - 8.29288f / (d + 5.52825f));
        phase["w_d"] = std::exp(-0.599085f / (d - 0.641583f) - 0.665888f);
    }

    void properties(Properties& props) override {
        props.config_float("particle size", particle_size_um,
                           "Droplet diameter in micrometer driving the Mie phase function (5 - 50).",
                           0.1f);
        props.config_float("max distance", max_distance,
                           "Distance at which the optical depth saturates.");
    }

    float3 mu_t{0.f};
    float3 mu_s{0.f};
    float max_distance = 1e7f;
    float particle_size_um = 7.f;
};

} // namespace merian
