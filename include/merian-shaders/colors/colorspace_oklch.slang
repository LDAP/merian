#pragma once

implementing colorspaces;

namespace merian {

// h in range 0, 2pi
public float3 oklch_to_rgb(float3 lch) {
    return oklab_to_rgb(float3(lch.x, lch.y * cos(lch.z), lch.y * sin(lch.z)));
}

public float3 rgb_to_oklch(float3 rgb) {
    float3 oklab = rgb_to_oklab(rgb);
    return float3(oklab.x, sqrt((oklab.y * oklab.y) * (oklab.z * oklab.z)),
                  atan2(oklab.z, oklab.y));
}

}
