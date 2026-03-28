struct CBA {
    float a;
    uint b;
};

struct CBB {
    int c;
    float d;
};

struct Params {
    ConstantBuffer<CBA> cb_a;
    ConstantBuffer<CBB> cb_b;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.output[0] = asuint(params.cb_a.a);
    params.output[1] = params.cb_a.b;
    params.output[2] = asuint(params.cb_b.c);
    params.output[3] = asuint(params.cb_b.d);
}
