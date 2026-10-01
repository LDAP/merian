namespace merian {

// Unit quaternions as float4(xyz, w).

float4 quaternion_identity() {
    return float4(0.f, 0.f, 0.f, 1.f);
}

// rotates by b, then by a
float4 quaternion_mul(const float4 a, const float4 b) {
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

float3 quaternion_rotate(const float4 q, const float3 v) {
    const float3 t = 2.f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// The shortest rotation from the unit vector from to the unit vector to; undefined for opposite
// vectors.
float4 quaternion_from_to(const float3 from, const float3 to) {
    return normalize(float4(cross(from, to), 1.f + dot(from, to)));
}

float4 quaternion_half_turn(const float3 axis) {
    return float4(axis, 0.f);
}

} // namespace merian
