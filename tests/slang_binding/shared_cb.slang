struct CBData {
    float x;
    int y;
};

struct ParamsA {
    ConstantBuffer<CBData> cb;
    RWStructuredBuffer<uint> output;
};

struct ParamsB {
    ConstantBuffer<CBData> cb;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<ParamsA> a, ParameterBlock<ParamsB> b) {
    a.output[0] = asuint(a.cb.x);
    a.output[1] = asuint(a.cb.y);
    b.output[0] = asuint(b.cb.x);
    b.output[1] = asuint(b.cb.y);
}
