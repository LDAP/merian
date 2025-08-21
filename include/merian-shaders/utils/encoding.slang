#pragma once

namespace merian {

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
float3 geo_decode_normal(uint enc) {
    float2 projected = unpackSnorm2x16ToFloat(enc);
    float3 vec = float3(projected, (1.0f - abs(projected.x)) - abs(projected.y));
    if (vec.z < 0.0f) {
        vec.xy = copysign(1.0 - abs(vec.yx), vec.xy);
    }
    return normalize(vec);
}

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
uint geo_encode_normal(float3 n) {
    float invL1Norm = 1.0f / ((abs(n.x) + abs(n.y)) + abs(n.z));
    float2 enc;
    if (n.z < 0.0f) {
        enc = copysign(1.0f - abs(n.yx * invL1Norm), n.xy);
    } else {
        enc = n.xy * invL1Norm;
    }
    return packSnorm2x16(enc);
}

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
float3 geo_decode_normal_float(float enc) {
    return geo_decode_normal(asuint(enc));
}

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
float geo_encode_normal_float(float3 n) {
    return asfloat(geo_encode_normal(n));
}

}
