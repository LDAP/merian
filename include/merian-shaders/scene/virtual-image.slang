import merian_shaders.scene.environment_map;
import merian_shaders.utils.quaternion;

namespace merian {

// Where the end of a chain of reflections and refractions appears to a viewer at its start.
// Sagittal and tangential paraxial transfer matrices (Coddington's equations), imaged at the mean
// vergence.
struct VirtualImage {
    static const float MIN_COS = 1e-4f;
    static const float MIN_DISTANCE = 1e-3f;
    static const float MIN_MAGNIFICATION = 1e-2f;

    float4 sagittal;
    float4 tangential;
    float4 unfold;
    bool mirrored;
    float travelled;

    __init() {
        sagittal = float4(1.f, 0.f, 0.f, 1.f);
        tangential = float4(1.f, 0.f, 0.f, 1.f);
        unfold = quaternion_identity();
        mirrored = false;
        travelled = 0.f;
    }

    [mutating]
    void propagate(const float distance) {
        sagittal.xy += distance * sagittal.zw;
        tangential.xy += distance * tangential.zw;
        travelled += distance;
    }

    // wi points back along the chain; curvatures are positive where the surface bulges towards wi
    [mutating]
    void reflect(const float3 wi, const float3 wo, const float2 curvature) {
        const float3 bisector = wi + wo;
        const float3 h = dot(bisector, bisector) > 0.f ? normalize(bisector) : wi;
        const float cos_i = max(dot(wi, h), MIN_COS);
        sagittal.zw += 2.f * curvature.x * cos_i * sagittal.xy;
        tangential.zw += 2.f * curvature.y / cos_i * tangential.xy;
        unfold = quaternion_mul(unfold, quaternion_half_turn(h));
        mirrored = !mirrored;
    }

    // eta is incident over transmitted
    [mutating]
    void refract(const float3 wi,
                 const float3 wo,
                 const float cos_incident,
                 const float cos_transmitted,
                 const float eta,
                 const float2 curvature) {
        const float cos_i = max(cos_incident, MIN_COS);
        const float cos_t = max(cos_transmitted, MIN_COS);
        const float2 power = (eta * cos_i - cos_t) * curvature;
        sagittal.zw = power.x * sagittal.xy + eta * sagittal.zw;
        const float2 ab = tangential.xy;
        tangential.zw = power.y / (cos_i * cos_t) * ab + eta * cos_i / cos_t * tangential.zw;
        tangential.xy = cos_t / cos_i * ab;
        unfold = quaternion_mul(unfold, quaternion_from_to(wo, -wi));
    }

    static float determinant(const float4 m) {
        return m.x * m.w - m.y * m.z;
    }

    float vergence() {
        return 0.5f * (sagittal.x / sagittal.y + tangential.x / tangential.y);
    }

    float distance() {
        const float v = vergence();
        return v > 0.f ? clamp(1.f / v, MIN_DISTANCE * travelled, ENV_MAP_DISTANCE) : travelled;
    }

    float3 displacement(const float3 end_displacement, const float3 direction) {
        const float axial = dot(end_displacement, direction);
        float3 d = end_displacement;
        if (vergence() > 0.f) {
            const float a = sign(sagittal.x) * sqrt(abs(sagittal.x * tangential.x));
            if (abs(a) > MIN_MAGNIFICATION) {
                const float det = sqrt(abs(determinant(sagittal) * determinant(tangential)));
                d = (end_displacement - axial * direction) / a + det / (a * a) * axial * direction;
            }
        }
        const float3 unfolded = quaternion_rotate(unfold, d);
        return mirrored ? -unfolded : unfolded;
    }
}

} // namespace merian
