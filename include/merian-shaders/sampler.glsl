#ifndef _MERIAN_SHADERS_SAMPLER_H_
#define _MERIAN_SHADERS_SAMPLER_H_

// Adapted for Vulkan from https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
// (MIT License)
//
// Samples a texture with Catmull-Rom filtering, using 9 texture fetches instead of 16.
// See http://vec3.ca/bicubic-filtering-in-fewer-taps/ for more details
vec4 catmull_rom(const sampler2D tex, const vec2 uv) {
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 texSize = textureSize(tex, 0);
    vec2 samplePos = uv * texSize;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
    vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    
    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 texPos0 = texPos1 - vec2(1.0);
    vec2 texPos3 = texPos1 + vec2(2.0);
    vec2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    vec4 result = vec4(0.0);
    result += textureLod(tex, vec2(texPos0.x,  texPos0.y),  0) * w0.x * w0.y;
    result += textureLod(tex, vec2(texPos12.x, texPos0.y),  0) * w12.x * w0.y;
    result += textureLod(tex, vec2(texPos3.x,  texPos0.y),  0) * w3.x * w0.y;

    result += textureLod(tex, vec2(texPos0.x,  texPos12.y), 0) * w0.x * w12.y;
    result += textureLod(tex, vec2(texPos12.x, texPos12.y), 0) * w12.x * w12.y;
    result += textureLod(tex, vec2(texPos3.x,  texPos12.y), 0) * w3.x * w12.y;

    result += textureLod(tex, vec2(texPos0.x,  texPos3.y),  0) * w0.x * w3.y;
    result += textureLod(tex, vec2(texPos12.x, texPos3.y),  0) * w12.x * w3.y;
    result += textureLod(tex, vec2(texPos3.x,  texPos3.y),  0) * w3.x * w3.y;

    return result;
}

// texel_coord in pixels
vec4 texelFetchClamp(const ivec2 texel_coord, sampler2D texture) {
    const ivec2 coord = clamp(texel_coord, ivec2(0), textureSize(texture, 0) - ivec2(1));
    return texelFetch(texture, coord, 0);
}

// Find longest motion vector inside a window, return motion vector in texture space (i.e. [0,1]^2)!
// 
// Reason:
// When reprojecting the current pixel, we need to realize that the velocity texture, unlike the history color buffer, is aliased.
// If we’re not careful we could be reintroducing edge aliasing indirectly. To better account for the edges, a typical solution is to dilate the aliased information.
// We’ll use velocity as an example but you can do this with depth and stencil. There are a couple of ways I know of doing it:
// 
// - (here:) Magnitude Dilation: take the velocity with the largest magnitude in a neighborhood
// - (also possible:) Depth Dilation: take the velocity that corresponds to the pixel with the nearest depth in a neighborhood
vec2 sample_motion_vector(const sampler2D img, const ivec2 center_pixel, const int radius) {
    // find longest motion vector, transform into texture space
    // access pixes position of the current fragment with ivec2(gl_FragCoord.xy)
    
    vec2 longest = vec2(0., 0.);
    for (int j = -radius; j <= radius; ++j) {
        for (int i = -radius; i <= radius; ++i) {
            const vec2 mv = texelFetchClamp(center_pixel + ivec2(j, i), img).rg;
            if (length(mv) > length(longest)) {
                longest = mv;
            }
        }
    }

    return longest;
}

#endif
