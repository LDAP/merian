#pragma once

namespace merian {

/**
 * Data used to create a material sample for a Material for a surface shading point or
 * volume point.
 */
struct ShadingData {
    float3 pos;
    float2 uv;
}

}
