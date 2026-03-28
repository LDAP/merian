struct Inner {
    float a;
    int b;
};

struct Params {
    Inner inner;
    uint c;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.output[0] = asuint(params.inner.a);
    params.output[1] = asuint(params.inner.b);
    params.output[2] = params.c;
}
