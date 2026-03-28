struct CBData {
    float val;
};

struct Params {
    ConstantBuffer<CBData> cb;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.output[0] = asuint(params.cb.val);
}
