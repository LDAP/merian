struct Params {
    float2 val_float2;
    float3 val_float3;
    float4 val_float4;
    int3 val_int3;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.output[0] = asuint(params.val_float2.x);
    params.output[1] = asuint(params.val_float2.y);
    params.output[2] = asuint(params.val_float3.x);
    params.output[3] = asuint(params.val_float3.y);
    params.output[4] = asuint(params.val_float3.z);
    params.output[5] = asuint(params.val_float4.x);
    params.output[6] = asuint(params.val_float4.y);
    params.output[7] = asuint(params.val_float4.z);
    params.output[8] = asuint(params.val_float4.w);
    params.output[9] = asuint(params.val_int3.x);
    params.output[10] = asuint(params.val_int3.y);
    params.output[11] = asuint(params.val_int3.z);
}
