import merian_shaders.gbuffer;
import merian_shaders.utils.textures;
import merian_shaders.colors.debug;

using merian;

struct GBufferDebugBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> output;
};

struct PushConstant {
    int selected_field;
};

[vk::push_constant]
ConstantBuffer<PushConstant> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID, ParameterBlock<GBufferDebugBinding> params) {

    uint2 pixel = tid.xy;
    if (not_on_texture(pixel, params.output))
        return;

    float4 color = float4(0, 0, 0, 1);

    switch (pc.selected_field) {
    case 0: { // Normal
        float3 n = params.gbuffer.get_normal(pixel);
        color = float4(debug_normal_color(n), 1);
        break;
    }
    case 1: { // Linear Z
        float z = params.gbuffer.get_linear_z(pixel);
        color = float4(float3(saturate(z / 100.0)), 1);
        break;
    }
    case 2: { // Grad Z
        float2 g = params.gbuffer.get_grad_z(pixel);
        color = float4(abs(g) * 10.0, 0, 1);
        break;
    }
    case 3: { // Delta Z
        float dz = params.gbuffer.get_delta_z(pixel);
        color = float4(saturate(-dz * 10), 0, saturate(dz * 10), 1);
        break;
    }
    case 4: { // Motion Vectors
        float2 m = params.gbuffer.get_motion_vectors(pixel);
        color = float4(abs(m) / 10.0, 0, 1);
        break;
    }
    case 5: { // Instance ID
        uint id = params.gbuffer.get_hit(pixel).instance_id;
        if (id == uint(-1)) {
            color = float4(0, 0, 0, 1);
        } else {
            color = float4(random_color(id), 1);
        }
        break;
    }
    case 6: { // Geometry Index
        uint id = params.gbuffer.get_hit(pixel).geometry_index;
        if (id == uint(-1)) {
            color = float4(0, 0, 0, 1);
        } else {
            color = float4(random_color(id), 1);
        }
        break;
    }
    case 7: { // Geometry ID
        uint id = params.gbuffer.get_hit(pixel).get_geometry_id();
        if (id == uint(-1)) {
            color = float4(0, 0, 0, 1);
        } else {
            color = float4(random_color(id), 1);
        }
        break;
    }
    case 8: { // Primitive ID
        uint id = params.gbuffer.get_hit(pixel).primitive_id;
        if (id == uint(-1)) {
            color = float4(0, 0, 0, 1);
        } else {
            color = float4(random_color(id), 1);
        }
        break;
    }
    case 9: { // Barycentrics
        float3 bary = params.gbuffer.get_hit(pixel).get_barycentrics3();
        color = float4(bary, 1);
        break;
    }
    case 10: { // Albedo
        color = float4(params.gbuffer.get_albedo(pixel).reflection, 1);
        break;
    }
    }

    params.output[pixel] = color;
}
