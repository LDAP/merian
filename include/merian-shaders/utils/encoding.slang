namespace merian {

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
float3 decode_normal(uint enc) {
    float2 projected = unpackSnorm2x16ToFloat(enc);
    float3 vec = float3(projected, (1.0f - abs(projected.x)) - abs(projected.y));
    if (vec.z < 0.0f) {
        vec.xy = copysign(1.0 - abs(vec.yx), vec.xy);
    }
    return normalize(vec);
}

// Octahedral projection of a unit vector onto [-1, 1]^2.
// A Survey of Efficient Representations for Independent Unit Vectors, JCGT Vol. 3, No. 2, 2014.
float2 octahedral_project(float3 n) {
    const float invL1Norm = 1.0f / ((abs(n.x) + abs(n.y)) + abs(n.z));
    if (n.z < 0.0f) {
        return copysign(1.0f - abs(n.yx * invL1Norm), n.xy);
    }
    return n.xy * invL1Norm;
}

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
uint encode_normal(float3 n) {
    return packSnorm2x16(octahedral_project(n));
}

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
float3 decode_normal_float(float enc) {
    return decode_normal(asuint(enc));
}

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
float encode_normal_float(float3 n) {
    return asfloat(encode_normal(n));
}

// LSB = handedness sign, upper 31 bits = oct direction. Worst-case angular error ~7.5e-5 rad.
uint encode_tangent(float4 t) {
    const uint dir = encode_normal(normalize(t.xyz)) & ~1u;
    return dir | (t.w < 0.f ? 1u : 0u);
}

float4 decode_tangent(uint enc) {
    const float w = (enc & 1u) != 0u ? -1.f : 1.f;
    return float4(decode_normal(enc & ~1u), w);
}

}
