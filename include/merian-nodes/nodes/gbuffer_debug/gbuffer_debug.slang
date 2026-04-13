import merian_shaders.gbuffer;
import merian_shaders.utils.textures;

using merian;

struct GBufferDebugBinding {
    Texture2D<uint4> tex0;
    Texture2D<uint4> tex1;
    Texture2D<float2> tex2;
    RWTexture2D<float4> output;
};

struct PushConstant {
    int selected_field;
};

[vk::push_constant]
ConstantBuffer<PushConstant> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID,
          ParameterBlock<GBufferDebugBinding> gbuffer) {

    uint2 pixel = tid.xy;
    if (not_on_texture(pixel, gbuffer.output))
        return;

    float4 color = float4(0, 0, 0, 1);

    switch (pc.selected_field) {
    case 0: { // Normal
        uint4 d = gbuffer.tex0[pixel];
        float3 n = decode_normal(d.x);
        color = float4(n * 0.5 + 0.5, 1);
        break;
    }
    case 1: { // Linear Z
        uint4 d = gbuffer.tex0[pixel];
        float z = asfloat(d.y);
        color = float4(float3(saturate(z / 100.0)), 1);
        break;
    }
    case 2: { // Grad Z
        uint4 d = gbuffer.tex0[pixel];
        float2 g = unpackHalf2x16ToFloat(d.z);
        color = float4(abs(g) * 10.0, 0, 1);
        break;
    }
    case 3: { // Delta Z
        uint4 d = gbuffer.tex0[pixel];
        float dz = asfloat(d.w);
        color = float4(saturate(-dz * 10), 0, saturate(dz * 10), 1);
        break;
    }
    case 4: { // Motion Vectors
        float2 m = gbuffer.tex2[pixel];
        color = float4(abs(m) / 10.0, 0, 1);
        break;
    }
    case 5: { // Instance ID
        uint4 h = gbuffer.tex1[pixel];
        uint id = h.x;
        if (id == uint(-1)) {
            color = float4(0, 0, 0, 1);
        } else {
            float r = frac(float(id) * 0.6180339887);
            float g = frac(float(id) * 0.3247179572);
            float b = frac(float(id) * 0.2122065908);
            color = float4(r, g, b, 1);
        }
        break;
    }
    case 6: { // Primitive ID
        uint4 h = gbuffer.tex1[pixel];
        uint id = h.y;
        if (id == uint(-1)) {
            color = float4(0, 0, 0, 1);
        } else {
            float r = frac(float(id) * 0.6180339887);
            float g = frac(float(id) * 0.3247179572);
            float b = frac(float(id) * 0.2122065908);
            color = float4(r, g, b, 1);
        }
        break;
    }
    case 7: { // Barycentrics
        uint4 h = gbuffer.tex1[pixel];
        float2 bary = unpackHalf2x16ToFloat(h.w);
        color = float4(bary.x, bary.y, 1.0 - bary.x - bary.y, 1);
        break;
    }
    }

    gbuffer.output[pixel] = color;
}
