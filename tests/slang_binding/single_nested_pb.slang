struct Inner {
    float a;
    int b;
    RWStructuredBuffer<uint> output;
};

struct Params {
    uint c;
    ParameterBlock<Inner> inner;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.inner.output[0] = asuint(params.inner.a);
    params.inner.output[1] = asuint(params.inner.b);
    params.inner.output[2] = params.c;
}
