#pragma once

import merian_shaders.utils.encoding;
import merian_shaders.utils.textures;
import merian_shaders.scene.scene;

namespace merian {

// Packed structs matching the texel layouts for reinterpret.

struct PackedSurfaceInfo {
    uint encoded_normal;
    float linear_z;
    half2 grad_z;
    float delta_z;
}

struct PackedHitInfo {
    uint instance_id;
    uint primitive_id;
    uint geometry_index;
    half2 barycentrics;

    static PackedHitInfo pack(Hit hit) {
        PackedHitInfo p;
        p.instance_id = hit.instance_id;
        p.primitive_id = hit.primitive_id;
        p.geometry_index = hit.geometry_index;
        p.barycentrics = half2(hit.barycentrics);
        return p;
    }

    Hit unpack() {
        return Hit(instance_id, primitive_id, geometry_index, float2(barycentrics));
    }
}

// Decoded surface data (batch-loaded from tex0).
struct SurfaceData {
    float3 normal;
    float linear_z;
    half2 grad_z;
    float delta_z;
};

// GBuffer with access-templated texture members.
//   tex0 (uint4):   .x = enc_normal, .y = linear_z, .z = grad_z, .w = delta_z
//   tex1 (uint4):   .x = instance_id, .y = primitive_id, .z = geometry_index, .w = barycentrics
//   tex2 (float2):  .xy = pixel-space motion vectors
struct __GBuffer<int access = 0> {
    RRWWTexture2D<uint4, access> tex0;
    RRWWTexture2D<uint4, access> tex1;
    RRWWTexture2D<float2, access> tex2;
}

// Read-only accessors
extension __GBuffer<0> {

    // Individual getters (each performs one texture load)
    float3 get_normal(uint2 pixel) { return decode_normal(tex0[pixel].x); }
    float get_linear_z(uint2 pixel) { return asfloat(tex0[pixel].y); }
    half2 get_grad_z(uint2 pixel) { return unpackHalf2x16ToHalf(tex0[pixel].z); }
    float get_delta_z(uint2 pixel) { return asfloat(tex0[pixel].w); }
    float2 get_motion_vectors(uint2 pixel) { return tex2[pixel]; }
    Hit get_hit(uint2 pixel) { return reinterpret<PackedHitInfo>(tex1[pixel]).unpack(); }

    // Batch loaders (single texture load, all fields decoded)
    SurfaceData load_surface(uint2 pixel) {
        uint4 raw = tex0[pixel];
        SurfaceData d;
        d.normal = decode_normal(raw.x);
        d.linear_z = asfloat(raw.y);
        d.grad_z = unpackHalf2x16ToHalf(raw.z);
        d.delta_z = asfloat(raw.w);
        return d;
    }

    Hit load_hit(uint2 pixel) { return get_hit(pixel); }
}

// Write-only accessors
extension __GBuffer<2> {

    void store_surface(uint2 pixel, float3 normal, float linear_z, half2 grad_z, float delta_z) {
        tex0.Store(pixel,
            reinterpret<uint4>(PackedSurfaceInfo(encode_normal(normal), linear_z, grad_z, delta_z)));
    }

    void store_hit(uint2 pixel, Hit hit) {
        tex1.Store(pixel, reinterpret<uint4>(PackedHitInfo::pack(hit)));
    }

    void store_motion_vectors(uint2 pixel, float2 mv) {
        tex2.Store(pixel, mv);
    }

    void clear(uint2 pixel) {
        tex0.Store(pixel, uint4(0));
        tex1.Store(pixel, uint4(uint(-1), uint(-1), uint(-1), 0));
        tex2.Store(pixel, float2(0));
    }
}

// read-only GBuffer
typealias GBuffer = __GBuffer<0>;

// TODO: Needs a way to 'combine' the specializations of 0 and 1
// read-write GBuffer
// typealias RWGBuffer = __GBuffer<1>;

// write-only GBuffer
typealias WGBuffer = __GBuffer<2>;

}
