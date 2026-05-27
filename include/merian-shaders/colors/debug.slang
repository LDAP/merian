import merian_shaders.utils.random;
import merian_shaders.colors.colorspaces;

namespace merian {

float3 debug_color(const float v) {
    if (isnan(v)) {
        return float3(1, 0, 0);
    } else if (isinf(v)) {
        return float3(0, 0, 1);
    } else if (v < 0) {
        return float3(0, -v, 0);
    }
    return float3(v);
}

float3 debug_normal_color(const float3 n) {
    if (length(n) < 1e-3) {
        return float3(0);
    }
    return n * 0.5 + 0.5;
}

float3 random_color(const uint v) {
    return float3(frac(float(v + 1) * 0.6180339887), frac(float(v + 1) * 0.3247179572),
                  frac(float(v + 1) * 0.2122065908));
}

}
