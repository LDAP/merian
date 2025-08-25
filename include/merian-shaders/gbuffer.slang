#pragma once

import merian_shaders.utils.encoding;

namespace merian {

typealias GBufferBinding = Texture2D<uint4>;
typealias RWGBufferBinding = RWTexture2D<uint4>;

struct CompressedGBuffer {
    uint _encoded_normal;
    float _linear_z;
    half2 _grad_z;
    float _delta_z;

    // surface normal
    float3 decompress_normal() {
        return merian::decode_normal(_encoded_normal);
    }

    // linear distance from camera to surface point
    float decompress_linear_z() {
        return _linear_z;
    }

    // Change of depth with respect to pixel difference
    // dlinear_z / dipos in depth / pixel
    half2 decompress_grad_z() {
        return _grad_z;
    }

    // Delta of distance to surface point compared to last frame
    float decompress_delta_z() {
        return _delta_z;
    }
}

struct GBuffer {
    static CompressedGBuffer load_compressed(GBufferBinding gbuffer_binding,
                                             const uint2 pixel) {
        return reinterpret<CompressedGBuffer>(gbuffer_binding.Load(uint3(pixel, 0)));
    }

    static GBuffer load(GBufferBinding gbuffer_binding, const uint2 pixel) {
        return GBuffer(reinterpret<CompressedGBuffer>(gbuffer_binding.Load(uint3(pixel, 0))));
    }

    // ------------------------------

    float3 normal;
    float linear_z;
    half2 grad_z;
    float delta_z;

    __init(const float3 normal, const float linear_z, const half2 grad_z, const float delta_z) {
        this.normal = normal;
        this.linear_z = linear_z;
        this.grad_z = grad_z;
        this.delta_z = delta_z;
    }

    __init(const CompressedGBuffer compressed) {
        this.normal = compressed.decompress_normal();
        this.linear_z = compressed.decompress_linear_z();
        this.grad_z = compressed.decompress_grad_z();
        this.delta_z = compressed.decompress_delta_z();
    }

    // surface normal
    float3 get_normal() {
        return normal;
    }

    // linear distance from camera to surface point
    float get_linear_z() {
        return linear_z;
    }

    // Change of depth with respect to pixel difference
    // dlinear_z / dipos in depth / pixel
    half2 get_grad_z() {
        return grad_z;
    }

    // Delta of distance to surface point compared to last frame
    float get_delta_z() {
        return delta_z;
    }
}

}
