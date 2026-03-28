struct CBData {
    float a;
    int b;
};

struct Params {
    float val;
    uint count;
    ConstantBuffer<CBData> cb;
    RWStructuredBuffer<uint> input;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.output[0] = asuint(params.val);
    params.output[1] = params.count;
    params.output[2] = asuint(params.cb.a);
    params.output[3] = asuint(params.cb.b);
    params.output[4] = params.input[0] + params.input[1];
}
