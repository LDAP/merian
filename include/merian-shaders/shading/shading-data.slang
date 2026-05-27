import merian_shaders.utils.frame;

namespace merian {

/**
 * Data used to create a material sample for a Material for a surface shading point or
 * volume point.
 */
struct ShadingData {
    float3 wi;  // the view direction (-ray.dir), pointing away from the surface (same as BSDF).
    float3 pos; // the current hit position
    float2 uv;  // the texture coordinates
    float time; // the current scene time
    OrthonormalFrame frame; // the (interpolated) frame
    float3 face_normal;     // geometric face normal in world space, on the mesh-canonical front side
    uint16_t bounce;        // 0 is the first surface (gbuffer)

    // True if wi is on the same side as face_normal (front-side hit).
    bool front_facing() {
        return dot(wi, face_normal) >= 0.f;
    }
}

}
