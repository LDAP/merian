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

    Albedo get_albedo(const float3 wi);
}

}
