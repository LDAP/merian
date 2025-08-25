#version 460
#extension GL_GOOGLE_include_directive    : enable
#extension GL_EXT_nonuniform_qualifier    : enable

#include "svgf_graph_layout.glsl"
#include "merian-shaders/sampler.glsl"
#include "merian-shaders/normal_encode.glsl"

layout(set = 1, binding = 0) uniform sampler2D img_filter_result;
layout(set = 1, binding = 1) uniform sampler2D img_filter_unused; // unused needed to match the filter layout
layout(set = 1, binding = 2) uniform usampler2D img_gbuf_result;
layout(set = 1, binding = 3) uniform usampler2D img_gbuf_unused;  // unused needed to match the filter layout

layout (constant_id = 2) const int DEBUG = 0;
layout (constant_id = 3) const int FILTER_PREV = 0;
layout (constant_id = 4) const int CLAMPING = 0;
layout (constant_id = 5) const int MV_SAMPLING = 0;
layout (constant_id = 6) const bool USE_MOTION_VECTORS = true;

layout(push_constant, std140) uniform params_t { 
    float blend_alpha;
    float rejection_threshold;
} params;

void
main()
{
  const ivec2 ipos = ivec2(gl_GlobalInvocationID);
  if (any(greaterThanEqual(ipos, imageSize(img_out)))) return;

  vec2 mv;
  if (USE_MOTION_VECTORS) {
    if (MV_SAMPLING == 0)
      mv = texelFetch(img_mv, ipos, 0).rg;
    else if (MV_SAMPLING == 1)
      mv = sample_motion_vector(img_mv, ipos, 1);
  } else {
    mv = vec2(0);
  }

  const vec4 filter_result = texelFetch(img_filter_result, ipos, 0);

  if (DEBUG == 1) {
    imageStore(img_out, ipos, vec4(filter_result.rgb, 1));
    return;
  }
  if (DEBUG == 2) {
    imageStore(img_out, ipos, vec4(filter_result.a / 256));
    return;
  }
  if (DEBUG == 3) {
    const vec3 normal = gbuffer_get_normal_pixel(gbuffer, ipos);
    imageStore(img_out, ipos, vec4(normal, 1));
    return;
  }
  if (DEBUG == 4) {
    imageStore(img_out, ipos, vec4(gbuffer_get_linear_z_pixel(gbuffer, ipos) / 1000));
    return;
  }
  if (DEBUG == 5) {
    imageStore(img_out, ipos, texelFetch(img_albedo, ipos, 0));
    return;
  }
  if (DEBUG == 6) {
    imageStore(img_out, ipos, vec4(gbuffer_get_grad_z_pixel(gbuffer, ipos), 0, 0));
    return;
  }
  if (DEBUG == 7) {
    imageStore(img_out, ipos, vec4(any(isnan(filter_result.rgb)), any(isinf(filter_result.rgb)), 0, 0));
    return;
  }
  if (DEBUG == 8) {
    const vec2 size = textureSize(img_mv, 0);
    imageStore(img_out, ipos, vec4(abs(texelFetch(img_mv, ipos, 0).rg) / max(size.x, size.y), 0, 0));
    return;
  }


  const vec3 irr = filter_result.rgb;
  const vec3 albedo = texelFetch(img_albedo, ipos, 0).rgb; // multiply albedo


  vec3 prev_out = vec3(0);
  if (FILTER_PREV == 0) {
    prev_out = texelFetch(img_prev_out, ivec2(round(ipos + mv)), 0).rgb;
  } else if (FILTER_PREV == 1) {
    prev_out = catmull_rom(img_prev_out, (ipos + mv + 0.5) / vec2(imageSize(img_out))).rgb;
  }

  if (any(isnan(prev_out)) || any(isinf(prev_out))) {
    prev_out = irr * albedo;
  }

  if (CLAMPING == 0) {
    // min max clamping
    vec3 neigh_clamp_min_color = vec3(1. / 0.);
    vec3 neigh_clamp_max_color = vec3(-1. / 0.);
    for (int j = -1; j <= 1; ++j) {
      for (int i = -1; i <= 1; ++i) {
        const vec3 a = texelFetch(img_albedo, ipos + ivec2(i, j), 0).rgb;
        const vec3 irr = texelFetch(img_filter_result, ipos + ivec2(i, j), 0).rgb;
        const vec3 color = (a * irr);
        neigh_clamp_min_color = min(neigh_clamp_min_color, color);
        neigh_clamp_max_color = max(neigh_clamp_max_color, color);
      }
    }
    prev_out = clamp(prev_out, neigh_clamp_min_color, neigh_clamp_max_color);
  } else if (CLAMPING == 1) {
    // moments clamping
    vec3 m1 = vec3(0.0f);
    vec3 m2 = vec3(0.0f);
    const int r = 1;
    for (int yy=-r;yy<=r;yy++)
      for(int xx=-r;xx<=r;xx++) {
        const vec3 a = texelFetch(img_albedo, ipos + ivec2(xx, yy), 0).rgb;
        const vec3 irr = texelFetch(img_filter_result, ipos + ivec2(xx, yy), 0).rgb;
        const vec3 b = (a * irr);
        m1 += b;
        m2 += b * b;
      }
    m1 /= (2.0 * r + 1) * (2.0 * r + 1);
    m2 /= (2.0 * r + 1) * (2.0 * r + 1);

    const vec3 sigma = sqrt(max(vec3(0), m2 - m1 * m1));
    const float thresh = params.rejection_threshold;
    prev_out = clamp(prev_out, max(vec3(0), m1 - thresh * sigma), m1 + thresh * sigma);
  }

  const vec3 beauty = irr * albedo;
  const vec3 blended = mix(beauty, prev_out, params.blend_alpha);

  imageStore(img_out, ipos, vec4(blended, filter_result.a));
}

