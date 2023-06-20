A generator node that pushes the Shadertoy variables as push constant.

Inputs:


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

void main()
{
  const uvec2 pixel = gl_GlobalInvocationID.xy;
  if((pixel.x >= iResolution.x) || (pixel.y >= iResolution.y))
  {
    return;
  }

  vec4 frag_color;
  mainImage(frag_color, pixel);
  // WebGL or Shadertoy does not do a Linear->sRGB conversion
  // thus the shader must output sRGB. But here the shader is expected to output
  // linear!
  imageStore(result, ivec2(pixel), toLinear(frag_color));
}
```
