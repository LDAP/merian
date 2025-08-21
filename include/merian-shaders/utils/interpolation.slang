#pragma once

namespace merian {

// like smoothstep but without Hermite smoothing
float linearstep(float edge0, float edge1, float x) {
    return saturate((x - edge0) / (edge1 - edge0));
}

}

