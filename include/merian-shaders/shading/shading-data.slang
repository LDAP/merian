import merian_shaders.utils.frame;

namespace merian {

// Per-interaction path state the integrator hands to the scene/material.
struct RenderContext {
    uint16_t bounce; // 0 is the first surface (gbuffer)
    float ior;       // index of refraction of the medium the ray is currently in

    __init() {
        bounce = 0;
        ior = 1.0f;
    }

    __init(const uint16_t bounce, const float ior) {
        this.bounce = bounce;
        this.ior = ior;
    }
}

/**
 * Data used to create a material sample for a Material for a surface shading point or
 * volume point.
 */
struct ShadingData {
    float3 wi;  // the view direction (-ray.dir), pointing away from the surface (same as BSDF).
    float3 pos; // the current hit position
    float2 uv;  // the texture coordinates
    float time; // the current scene time
    OrthonormalFrame frame; // shading frame; flipped for backfacing to match the incoming ray dir
    float3 face_normal;     // geometric face normal; flipped for backfacing to match the incoming ray dir
    uint16_t bounce;        // 0 is the first surface (gbuffer)
    float ior;              // index of refraction of the medium the ray is currently in
    bool front_facing;      // true if wi hit the mesh-canonical front side (i.e. an entering ray)
}

}
