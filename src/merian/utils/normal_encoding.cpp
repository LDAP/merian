#include "merian/utils/normal_encoding.hpp"

#include <math.h>

#define CLAMP(_minval, x, _maxval) ((x) < (_minval) ? (_minval) : (x) > (_maxval) ? (_maxval) : (x))

namespace merian {

uint32_t encode_normal(float vec[3]) noexcept {
    uint32_t result = 0;
    int16_t* enc = reinterpret_cast<int16_t*>(&result);
    const float invL1Norm = 1.0f / (fabsf(vec[0]) + fabsf(vec[1]) + fabsf(vec[2]));
    // first find floating point values of octahedral map in [-1,1]:
    float enc0, enc1;
    if (vec[2] < 0.0f) {
        enc0 = (1.0f - fabsf(vec[1] * invL1Norm)) * ((vec[0] < 0.0f) ? -1.0f : 1.0f);
        enc1 = (1.0f - fabsf(vec[0] * invL1Norm)) * ((vec[1] < 0.0f) ? -1.0f : 1.0f);
    } else {
        enc0 = vec[0] * invL1Norm;
        enc1 = vec[1] * invL1Norm;
    }
    enc[0] = roundf(CLAMP(-32768.0f, enc0 * 32768.0f, 32767.0f));
    enc[1] = roundf(CLAMP(-32768.0f, enc1 * 32768.0f, 32767.0f));
    return result;
}

uint32_t encode_normal(float3 vec) noexcept {
    return encode_normal(&vec.x);
}

float3 decode_normal(uint32_t enc) noexcept {
    const int16_t* p = reinterpret_cast<const int16_t*>(&enc);
    float x = p[0] / 32768.0f;
    float y = p[1] / 32768.0f;
    float z = 1.0f - fabsf(x) - fabsf(y);
    if (z < 0.0f) {
        const float ox = (1.0f - fabsf(y)) * (x < 0.0f ? -1.0f : 1.0f);
        const float oy = (1.0f - fabsf(x)) * (y < 0.0f ? -1.0f : 1.0f);
        x = ox;
        y = oy;
    }
    const float len = sqrtf(x * x + y * y + z * z);
    return float3(x / len, y / len, z / len);
}

uint32_t encode_tangent(float4 t) noexcept {
    const float l = sqrtf(t.x * t.x + t.y * t.y + t.z * t.z);
    const float3 n = (l > 0.f) ? float3(t.x / l, t.y / l, t.z / l) : float3(0, 0, 1);
    const uint32_t dir = encode_normal(n) & ~1u;
    return dir | (t.w < 0.f ? 1u : 0u);
}

float4 decode_tangent(uint32_t enc) noexcept {
    const float w = (enc & 1u) ? -1.f : 1.f;
    const float3 dir = decode_normal(enc & ~1u);
    return float4(dir.x, dir.y, dir.z, w);
}

} // namespace merian
