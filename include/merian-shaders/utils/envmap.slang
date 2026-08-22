#include "merian-shaders/utils/math-constants.slangh"

namespace merian {

// Axis aligned cubemap indices
// 0: pos x, 1: neg x, 2: pos y, 3: neg: y, 4: pos z, 5: neg z
uint cubemap_side(const float3 w) {
    // if (abs(w.x) > abs(w.y) && abs(w.x) > abs(w.z)) {
    //     return w.x >= 0 ? 0 : 1;
    // } else if (abs(w.y) > abs(w.x) && abs(w.y) > abs(w.z)) {
    //     return w.y >= 0 ? 2 : 3;
    // } else {
    //     return w.z >= 0 ? 4 : 5;
    // }
    
    // branchless version of the above:

    const float3 a = abs(w);                                                                                   
    const bool xd = a.x > a.y && a.x > a.z, yd = a.y > a.z;                                                  
    return (xd ? 0u : yd ? 2u : 4u) + uint((xd ? w.x : yd ? w.y : w.z) < 0.0); 
}

// 45 deg turned cubemap indices
uint cubemap_side_45deg(const float3 w) {
    const float s = w.x + w.y;
    const float t = MERIAN_M_SQRT2 * w.z;
    const float3 d = float3(MERIAN_M_SQRT2 * (w.x - w.y), s - t, s + t);
    const float3 a = abs(d);
    const bool xd = a.x > a.y && a.x > a.z, yd = a.y > a.z;
    return (xd ? 0u : yd ? 2u : 4u) + uint((xd ? d.x : yd ? d.y : d.z) < 0.0);
}

}

