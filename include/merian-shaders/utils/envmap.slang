#pragma once

namespace merian {

// 0: pos x, 1: neg x, 2: pos y, 3: neg: y, 4: pos z, 5: neg z
uint cubemap_side(const float3 w) {
    if (abs(w.x) > abs(w.y) && abs(w.x) > abs(w.z)) {
        return w.x >= 0 ? 0 : 1;
    } else if (abs(w.y) > abs(w.x) && abs(w.y) > abs(w.z)) {
        return w.y >= 0 ? 2 : 3;
    } else {
        return w.z >= 0 ? 4 : 5;
    }
}

}

