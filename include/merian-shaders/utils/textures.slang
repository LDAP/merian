#pragma once

namespace merian {

uint texture_dimensions<T : ITexelElement, int sampleCount, int access, int isCombined, int format>(
    const _Texture<T, __Shape1D, 0, 0, sampleCount, access, 0, isCombined, format> tex,
    const int level) {
    uint dim;
    uint dummy;
    tex.GetDimensions(level, dim, dummy);
    return dim;
}

uint texture_dimensions<T : ITexelElement, int sampleCount, int access, int isCombined, int format>(
    const _Texture<T, __Shape1D, 0, 0, sampleCount, access, 0, isCombined, format> tex) {
    uint dim;
    tex.GetDimensions(dim);
    return dim;
}

uint2 texture_dimensions<T : ITexelElement,
                         int sampleCount,
                         int access,
                         int isCombined,
                         int format>(
    const _Texture<T, __Shape2D, 0, 0, sampleCount, access, 0, isCombined, format> tex,
    const int level) {
    uint2 dims;
    uint dummy;
    tex.GetDimensions(level, dims.x, dims.y, dummy);
    return dims;
}

uint2 texture_dimensions<T : ITexelElement,
                         int sampleCount,
                         int access,
                         int isCombined,
                         int format>(
    const _Texture<T, __Shape2D, 0, 0, sampleCount, access, 0, isCombined, format> tex) {
    uint2 dims;
    tex.GetDimensions(dims.x, dims.y);
    return dims;
}

uint3 texture_dimensions<T : ITexelElement,
                         int sampleCount,
                         int access,
                         int isCombined,
                         int format>(
    const _Texture<T, __Shape3D, 0, 0, sampleCount, access, 0, isCombined, format> tex,
    const int level) {
    uint3 dims;
    uint dummy;
    tex.GetDimensions(level, dims.x, dims.y, dims.z, dummy);
    return dims;
}

uint3 texture_dimensions<T : ITexelElement,
                         int sampleCount,
                         int access,
                         int isCombined,
                         int format>(
    const _Texture<T, __Shape3D, 0, 0, sampleCount, access, 0, isCombined, format> tex) {
    uint3 dims;
    tex.GetDimensions(dims.x, dims.y, dims.z);
    return dims;
}

bool is_on_texture<T : ITexelElement, int sampleCount, int access, int isCombined, int format>(
    const int2 pixel,
    const _Texture<T, __Shape2D, 0, 0, sampleCount, access, 0, isCombined, format> tex,
    const int level) {
    if (any(pixel < 0))
        return false;
    uint2 dims;
    uint dummy;
    tex.GetDimensions(level, dims.x, dims.y, dummy);
    return all(pixel < dims);
}

bool is_on_texture<T : ITexelElement, int sampleCount, int access, int isCombined, int format>(
    const int2 pixel,
    const _Texture<T, __Shape2D, 0, 0, sampleCount, access, 0, isCombined, format> tex) {
    if (any(pixel < 0))
        return false;
    uint2 dims;
    tex.GetDimensions(dims.x, dims.y);
    return all(pixel < dims);
}

bool not_on_texture<T : ITexelElement, int sampleCount, int access, int isCombined, int format>(
    const int2 pixel,
    const _Texture<T, __Shape2D, 0, 0, sampleCount, access, 0, isCombined, format> tex,
    const int level) {
    if (any(pixel < 0))
        return true;
    uint2 dims;
    uint dummy;
    tex.GetDimensions(level, dims.x, dims.y, dummy);
    return any(pixel >= dims);
}

bool not_on_texture<T : ITexelElement, int sampleCount, int access, int isCombined, int format>(
    const int2 pixel,
    const _Texture<T, __Shape2D, 0, 0, sampleCount, access, 0, isCombined, format> tex) {
    if (any(pixel < 0))
        return true;
    uint2 dims;
    tex.GetDimensions(dims.x, dims.y);
    return any(pixel >= dims);
}

// Adapted for Vulkan from https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
// (MIT License)
//
// Samples a texture with Catmull-Rom filtering, using 9 texture fetches instead of 16.
// See http://vec3.ca/bicubic-filtering-in-fewer-taps/ for more details
float4 catmull_rom(Sampler2D<float4> tex, float2 uv) {
    float2 texSize = float2(texture_dimensions(tex, uint(0)));

    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do
    // this by rounding down the sample location to get the exact center of our "starting" texel.
    // The starting texel will be at location [1, 1] in the grid, where [0, 0] is the top left
    // corner.
    float2 samplePos = uv * texSize;
    float2 texPos1 = floor(samplePos - 0.5f.xx) + 0.5f.xx;

    // Compute the fractional offset from our starting texel to our original sample location, which
    // we'll feed into the Catmull-Rom spline function to get our filter weights.
    float2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    float2 w12 = w1 + w2;
    float2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    float2 texPos0 = texPos1 - 1.0f.xx;
    float2 texPos3 = texPos1 + 2.0f.xx;
    float2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    float4 result = 0.0f;
    result += (tex.SampleLevel(float2(texPos0.x, texPos0.y), 0.0f) * w0.x) * w0.y;
    result += (tex.SampleLevel(float2(texPos12.x, texPos0.y), 0.0f) * w12.x) * w0.y;
    result += (tex.SampleLevel(float2(texPos3.x, texPos0.y), 0.0f) * w3.x) * w0.y;
    result += (tex.SampleLevel(float2(texPos0.x, texPos12.y), 0.0f) * w0.x) * w12.y;
    result += (tex.SampleLevel(float2(texPos12.x, texPos12.y), 0.0f) * w12.x) * w12.y;
    result += (tex.SampleLevel(float2(texPos3.x, texPos12.y), 0.0f) * w3.x) * w12.y;
    result += (tex.SampleLevel(float2(texPos0.x, texPos3.y), 0.0f) * w0.x) * w3.y;
    result += (tex.SampleLevel(float2(texPos12.x, texPos3.y), 0.0f) * w12.x) * w3.y;
    result += (tex.SampleLevel(float2(texPos3.x, texPos3.y), 0.0f) * w3.x) * w3.y;

    return result;
}

// texel_coord in pixels
T texelFetchClamp<T : ITexelElement, int sampleCount, int isCombined, int format>(
    int2 texel_coord, _Texture<T, __Shape2D, 0, 0, sampleCount, 0, 0, isCombined, format> tex) {
    int2 coord =
        clamp(texel_coord, int2(0, 0), int2(texture_dimensions(tex, uint(0))) - int2(1, 1));
    return tex.Load(int3(coord, 0));
}

// Find longest motion vector inside a window, return motion vector in texture space (i.e. [0,1]^2)!
//
// Reason:
// When reprojecting the current pixel, we need to realize that the velocity texture, unlike the
// history color buffer, is aliased. If we're not careful we could be reintroducing edge aliasing
// indirectly. To better account for the edges, a typical solution is to dilate the aliased
// information. We'll use velocity as an example but you can do this with depth and stencil. There
// are a couple of ways I know of doing it:
//
// - (here:) Magnitude Dilation: take the velocity with the largest magnitude in a neighborhood
// - (also possible:) Depth Dilation: take the velocity that corresponds to the pixel with the
// nearest depth in a neighborhood
float2 sample_motion_vector<int sampleCount, int isCombined, int format>(
    _Texture<float4, __Shape2D, 0, 0, sampleCount, 0, 0, isCombined, format> tex,
    int2 center_pixel,
    int radius) {
    // find longest motion vector, transform into texture space
    // access pixel position of the current fragment with ivec2(gl_FragCoord.xy)

    float2 longest = float2(0.0f);
    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            float2 mv = texelFetchClamp(center_pixel + int2(j, i), tex).xy;
            if (length(mv) > length(longest)) {
                longest = mv;
            }
        }
    }
    return longest;
}

int2 texturewrap_clamp(int2 ipos, int2 tex_size) {
    return clamp(ipos, int2(0, 0), tex_size - 1);
}

int2 texturewrap_repeat(int2 ipos, int2 tex_size) {
    return ipos % tex_size;
}

// mirror(a) returns a if a ≥ 0, and −(1 + a) otherwise
int2 mirror(int2 ipos) {
    return select(ipos < 0, -(1 + ipos), ipos);
}

int2 texturewrap_mirror_repeat(int2 ipos, int2 tex_size) {
    return tex_size - 1 - mirror(ipos % (tex_size * 2) - tex_size);
}

float2 pixel_to_uv_position(const uint2 pixel, const uint2 frame_dimensions) {
    return (pixel + 0.5f) / frame_dimensions;
}

float2 pixel_to_uv_position(const float uv, const uint2 frame_dimensions) {
    return uv * frame_dimensions - 0.5;
}

float2 pixel_to_ndc(const uint2 pixel, const uint2 frame_dimensions) {
    return float2(2, -2) * pixel_to_uv_position(pixel, frame_dimensions) + float2(-1, 1);
}

interface LODSampler {

    // LOD Sample method for a combined image sampler (SamplerXD).
    [ForceInline]
    T sample<T : ITexelElement,
             Shape : __ITextureShape,
             int isArray,
             int isMS,
             int sampleCount,
             int isShadow,
             int format>(
        const _Texture<T, Shape, isArray, isMS, sampleCount, 0, isShadow, 1, format> tex,
        const vector<float, Shape.dimensions + isArray> location);

    // LOD Sample method for a texture and sampler.
    [ForceInline]
    T sample<T : ITexelElement,
             Shape : __ITextureShape,
             int isArray,
             int isMS,
             int sampleCount,
             int isShadow,
             int format>(
        const _Texture<T, Shape, isArray, isMS, sampleCount, 0, isShadow, 0, format> tex,
        const SamplerState s,
        const vector<float, Shape.dimensions + isArray> location);
}

// Uses the textures ::Sample(...) method. LOD is computed using screen space derivatives, in OpenGL
// terms this is called implicit LOD calculation.
struct ImplicitLODSampler : LODSampler {

    [ForceInline]
    T sample<T : ITexelElement,
             Shape : __ITextureShape,
             int isArray,
             int isMS,
             int sampleCount,
             int isShadow,
             int format>(
        const _Texture<T, Shape, isArray, isMS, sampleCount, 0, isShadow, 1, format> tex,
        const vector<float, Shape.dimensions + isArray> location) {
        return tex.Sample(location);
    }

    [ForceInline]
    T sample<T : ITexelElement,
             Shape : __ITextureShape,
             int isArray,
             int isMS,
             int sampleCount,
             int isShadow,
             int format>(
        const _Texture<T, Shape, isArray, isMS, sampleCount, 0, isShadow, 0, format> tex,
        const SamplerState s,
        const vector<float, Shape.dimensions + isArray> location) {
        return tex.Sample(s, location);
    }
}

typealias DefaultLODSampler = ImplicitLODSampler;

// Uses the textures ::SampleLod(...) method. In OpenGL terms this is called explicit LOD.
struct ExplicitLODSampler : LODSampler {
    float lod;

    __init(const float lod) {
        this.lod = lod;
    }

    [ForceInline]
    T sample<T : ITexelElement,
             Shape : __ITextureShape,
             int isArray,
             int isMS,
             int sampleCount,
             int isShadow,
             int format>(
        const _Texture<T, Shape, isArray, isMS, sampleCount, 0, isShadow, 1, format> tex,
        const vector<float, Shape.dimensions + isArray> location) {
        return tex.SampleLevel(location, lod);
    }

    [ForceInline]
    T sample<T : ITexelElement,
             Shape : __ITextureShape,
             int isArray,
             int isMS,
             int sampleCount,
             int isShadow,
             int format>(
        const _Texture<T, Shape, isArray, isMS, sampleCount, 0, isShadow, 0, format> tex,
        const SamplerState s,
        const vector<float, Shape.dimensions + isArray> location) {
        return tex.SampleLevel(s, location, lod);
    }
}

// Uses the textures ::SampleLod(...) method to always sample LOD 0.
struct LOD0Sampler : LODSampler {

    [ForceInline]
    T sample<T : ITexelElement,
             Shape : __ITextureShape,
             int isArray,
             int isMS,
             int sampleCount,
             int isShadow,
             int format>(
        const _Texture<T, Shape, isArray, isMS, sampleCount, 0, isShadow, 1, format> tex,
        const vector<float, Shape.dimensions + isArray> location) {
        return tex.SampleLevel(location, 0);
    }

    [ForceInline]
    T sample<T : ITexelElement,
             Shape : __ITextureShape,
             int isArray,
             int isMS,
             int sampleCount,
             int isShadow,
             int format>(
        const _Texture<T, Shape, isArray, isMS, sampleCount, 0, isShadow, 0, format> tex,
        const SamplerState s,
        const vector<float, Shape.dimensions + isArray> location) {
        return tex.SampleLevel(s, location, 0);
    }
}

// Uses the textures ::SampleGrad(...) method to sample a LOD using explicit gradients.
struct GradLODSampler : LODSampler {
    float grad_x;
    float grad_y;

    __init(float grad_x, float grad_y) {
        this.grad_x = grad_x;
        this.grad_y = grad_y;
    }

    [ForceInline]
    T sample<T : ITexelElement,
             Shape : __ITextureShape,
             int isArray,
             int isMS,
             int sampleCount,
             int isShadow,
             int format>(
        const _Texture<T, Shape, isArray, isMS, sampleCount, 0, isShadow, 1, format> tex,
        const vector<float, Shape.dimensions + isArray> location) {
        return tex.SampleGrad(location, grad_x, grad_y);
    }

    [ForceInline]
    T sample<T : ITexelElement,
             Shape : __ITextureShape,
             int isArray,
             int isMS,
             int sampleCount,
             int isShadow,
             int format>(
        const _Texture<T, Shape, isArray, isMS, sampleCount, 0, isShadow, 0, format> tex,
        const SamplerState s,
        const vector<float, Shape.dimensions + isArray> location) {
        return tex.SampleGrad(s, location, grad_x, grad_y);
    }
}

}
