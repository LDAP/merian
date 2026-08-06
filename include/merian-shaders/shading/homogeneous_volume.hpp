#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/utils/vector_matrix.hpp"

#include <limits>

namespace merian {

// Mirrors the Slang HomogeneousVolume in shading/homogeneous-volume.slang.
struct HomogeneousVolume {
    float3 mu_t{0.f};
    float3 mu_s{0.f};
    float eta = 1.f;
    float particle_size_um = 25.f;
    float max_distance = std::numeric_limits<float>::max();

    bool is_vacuum() const {
        return mu_t.x == 0.f && mu_t.y == 0.f && mu_t.z == 0.f;
    }

    bool operator==(const HomogeneousVolume& other) const = default;

    void write_to(ShaderCursor cursor) const {
        cursor["mu_t"] = mu_t;
        cursor["mu_s"] = mu_s;
        cursor["eta"] = eta;
        cursor["particle_size_um"] = particle_size_um;
        cursor["max_distance"] = max_distance;
    }
};

} // namespace merian
