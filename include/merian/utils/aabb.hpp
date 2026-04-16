#pragma once

#include "merian/utils/vector_matrix.hpp"
namespace merian {

class AABB {
  public:
    AABB(const float3& box_min, const float3& box_max) : b_min(box_min), b_max(box_max) {}

    // Initializes AABB that is [max,lowest] to be expanded
    AABB() {
        reset();
    }

    const float3& get_min() const {
        return b_min;
    }

    const float3& get_max() const {
        return b_max;
    }

    float3 get_center() const {
        return (b_min + b_max) / 2.f;
    }

    float3 get_corner(const uint32_t i) const {
        assert(i < 8);

        float3 corner;
        corner.x = ((i & 0b001) != 0u) ? b_max.x : b_min.x;
        corner.y = ((i & 0b010) != 0u) ? b_max.y : b_min.y;
        corner.z = ((i & 0b100) != 0u) ? b_max.z : b_min.z;

        return corner;
    }

    bool is_valid() const {
        return all(lessThanEqual(b_min, b_max));
    }

    void expand(const float3& v) {
        b_min = min(v, b_min);
        b_max = max(v, b_max);
    }

    // Resets the AABB to [max,lowest] to be expanded
    void reset() {
        b_min = float3(std::numeric_limits<float>::max());
        b_max = float3(std::numeric_limits<float>::lowest());
    }

  private:
    float3 b_min;
    float3 b_max;
};

} // namespace merian
