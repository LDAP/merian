#pragma once

__exported import merian_shaders.shading.shading_function;

namespace merian {

struct Albedo {
    float3 reflection;

    __init(const float3 reflected) {
        reflection = reflected;
    }
}

static const float BSDF_MIN_COS_THETA = 1e-6f;

interface BSDF : ShadingFunction {

    // The incident direction points away from the
    // shading location.
    Albedo get_albedo(const float3 wi);

    // Pre-integrated specular reflectance for the given F0.
    float3 get_preintegrated_specular(const float3 F0, const float3 wi);
}

}
