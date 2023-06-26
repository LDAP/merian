#include "normal_encoding.hpp"

#include <math.h>

#define CLAMP(_minval, x, _maxval) ((x) < (_minval) ? (_minval) : (x) > (_maxval) ? (_maxval) : (x))

namespace merian {

uint32_t encode_normal(float* vec) {
    uint32_t result;
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

uint32_t encode_normal(glm::vec3 vec) {
    return encode_normal(&vec.x);
}

} // namespace merian
