struct CBData {
    float x;
    int y;
    float3 z;
};

struct Params {
    ConstantBuffer<CBData> cb;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.output[0] = asuint(params.cb.x);
    params.output[1] = asuint(params.cb.y);
    params.output[2] = asuint(params.cb.z.x);
    params.output[3] = asuint(params.cb.z.y);
    params.output[4] = asuint(params.cb.z.z);
}
