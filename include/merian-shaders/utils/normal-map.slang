namespace merian {

// Keller et al. [2017] workaround for artifacts.
// wi pointing away from the surface point.
float3 keller_clamp_normal(const float3 shading_normal, const float3 face_normal, const float3 wi) {
    const float3 r = reflect(-wi, shading_normal);
    if (dot(r, face_normal) < 0) {
        return normalize(wi + normalize(r - face_normal * dot(face_normal, r)));
    }
    return shading_normal;
}

}
