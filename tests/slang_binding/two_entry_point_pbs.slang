struct ParamsA {
    float a;
    uint b;
    RWStructuredBuffer<uint> output;
};

struct ParamsB {
    int c;
    float d;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<ParamsA> pa, ParameterBlock<ParamsB> pb) {
    pa.output[0] = asuint(pa.a);
    pa.output[1] = pa.b;
    pa.output[2] = asuint(pb.c);
    pa.output[3] = asuint(pb.d);
}
