A generator node that pushes the Shadertoy variables as push constant.

Outputs:

| Type  | Input ID | Input name | Description          | Format/Resolution            | Persistent |
|-------|----------|------------|----------------------|------------------------------|------------|
| Image | 0        | out        | the shadertoy result | `vk::Format::eR8G8B8A8Unorm` | no         |


Shader code:

```c++
#version 460

layout(local_size_x_id = 0, local_size_y_id = 1) in;
layout(binding = 0, set = 0) uniform writeonly image2D result;
layout(push_constant) uniform constants {
    vec2 iResolution;
    float iTime;
    float iTimeDelta;
    float iFrame;
};

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // your shadertoy code goes here
}

vec4 toLinear(vec4 sRGB) {
    bvec4 cutoff = lessThan(sRGB, vec4(0.04045));
    vec4 higher = pow((sRGB + vec4(0.055))/vec4(1.055), vec4(2.4));
    vec4 lower = sRGB/vec4(12.92);

    return mix(higher, lower, cutoff);
}

void main() {
  const uvec2 pixel = gl_GlobalInvocationID.xy;
  if((pixel.x >= iResolution.x) || (pixel.y >= iResolution.y)) {
    return;
  }

  vec4 frag_color;
  // In OpenGL the y axis is flipped
  mainImage(frag_color, ivec2(pixel.x, iResolution.y - pixel.y));
  // WebGL or Shadertoy does not do a Linear->sRGB conversion
  // thus the shader must output sRGB. But here the shader is expected to output
  // linear!
  imageStore(result, ivec2(pixel), toLinear(frag_color));
}
```
