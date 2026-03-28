namespace merian {

   // TODO:
    // https://github.com/NVIDIAGameWorks/Falcor/blob/eb540f6748774680ce0039aaf3ac9279266ec521/Source/RenderPasses/GBuffer/GBuffer/GBufferRT.slang#L86

// ---------------------------------------------------------------
// Value types (embedded in parent's ordinary data buffer)

struct MaterialParams {
    float roughness;
    float metallic;
};

// ---------------------------------------------------------------
// CB inner types (each gets its own uniform buffer in parent's descriptor set)

struct LightInfo {
    float3 direction;
    float intensity;
};

struct NestedLightInfo {
    float3 color;
    float power;
};

struct DeepLightInfo {
    float3 ambient;
    float strength;
};

// Resource-only struct: no ordinary data, no constant buffer — only a descriptor
struct TextureOnly {
    Texture2D<float4> tex;
};

// Test: ConstantBuffer containing another ConstantBuffer
struct InnerCB {
    float inner_val;
};

struct OuterCB {
    float outer_val;
    ConstantBuffer<InnerCB> inner;
};

// ---------------------------------------------------------------
// Test: explicit ShaderObject creation + assignment via write(ShaderObjectHandle)

struct ManualCBData {
    float manual_cb_val;
};

struct ManualPBData {
    float manual_pb_val;
    Texture2D<float4> manual_pb_tex;
};

// Test: reassignment — this CB will be replaced with a new object mid-run
struct ReplaceCBData {
    float replace_val;
};

// ---------------------------------------------------------------
// Level 3: deepest PB — value only
struct DeeperParams {
    float deepest_value;
};

// Level 2: PB inside PB — has value + CB + PB (all three binding modes)
struct DeepParams {
    float deep_value;                               // value
    ConstantBuffer<DeepLightInfo> deep_light;       // CB at depth 2
    ParameterBlock<DeeperParams> deeper;            // PB at depth 3
};

// Level 1: nested PB — has resource + value + PB + CB (all binding modes)
struct NestedParams {
    [[vk::binding(1, 1)]]
    Texture2D<float4> extra_map;                    // resource
    float weight;                                   // value
    ParameterBlock<DeepParams> deep;                // PB at depth 2
    [[vk::binding(2, 1)]]
    ConstantBuffer<NestedLightInfo> nested_light;   // CB at depth 1
};

// Level 0: root PB — has value + CB + PB + resources + resource-only PB + manual assignment tests
struct GBufferParams {
    MaterialParams material;                        // value (embedded struct)
    [[vk::binding(1, 0)]]
    ConstantBuffer<LightInfo> light;                // CB at depth 0
    [[vk::binding(4, 0)]]
    ConstantBuffer<OuterCB> cb_in_cb;               // CB containing CB
    ParameterBlock<NestedParams> nested;            // PB at depth 1
    ParameterBlock<TextureOnly> tex_only;           // PB with no ordinary data
    ConstantBuffer<ManualCBData> manual_cb;         // explicit ShaderObject assignment (CB)
    ParameterBlock<ManualPBData> manual_pb;         // explicit ShaderObject assignment (PB)
    ConstantBuffer<ReplaceCBData> replace_cb;       // reassignment test
    [[vk::binding(2, 0)]]
    Texture2D<float4> input;                        // resource
    [[vk::binding(3, 0)]]
    RWTexture2D<uint4> output;                      // resource (storage)
};

// ---------------------------------------------------------------
// Nesting structure (5 descriptor sets):
//   set 0: GBufferParams     — material(value), light(CB), input/output(resource)
//   set 1: NestedParams      — extra_map(resource), weight(value), nested_light(CB)
//   set 2: DeepParams        — deep_value(value), deep_light(CB)
//   set 3: DeeperParams      — deepest_value(value)
//
// Each level tests: value binding, CB staging upload, PB recursive bind.

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID, ParameterBlock<GBufferParams> params) {
    uint2 pixel = tid.xy;

    uint2 dims;
    params.output.GetDimensions(dims.x, dims.y);
    if (any(pixel >= dims))
        return;

    // Verify data from every binding mode at every nesting level.
    // Output channels encode sums so we can verify all paths carry data.
    params.output[pixel] = uint4(
        // Channel 0: value bindings at all levels + explicit CB/PB assignment + reassignment
        asuint(params.material.roughness
             + params.nested.weight
             + params.nested.deep.deep_value
             + params.nested.deep.deeper.deepest_value
             + params.manual_cb.manual_cb_val
             + params.manual_pb.manual_pb_val
             + params.replace_cb.replace_val),

        // Channel 1: CB bindings at all levels (light.intensity updated per-frame)
        asuint(params.light.intensity
             + params.nested.nested_light.power
             + params.nested.deep.deep_light.strength),

        // Channel 2: deep_value (updated per-frame) + manual PB texture sample
        asuint(params.nested.deep.deep_value)
        + params.manual_pb.manual_pb_tex[pixel].r,

        // Channel 3: deepest value + resource-only PB sample + CB-in-CB test
        asuint(params.nested.deep.deeper.deepest_value)
        + params.tex_only.tex[pixel].r
        + asuint(params.cb_in_cb.outer_val + params.cb_in_cb.inner.inner_val)
    );
}

}
