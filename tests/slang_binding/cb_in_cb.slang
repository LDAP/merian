struct Inner {
    float a;
    int b;
};

struct Outer {
    ConstantBuffer<Inner> inner;
    uint c;
};

struct Params {
    ConstantBuffer<Outer> outer;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.output[0] = asuint(params.outer.inner.a);
    params.output[1] = asuint(params.outer.inner.b);
    params.output[2] = params.outer.c;
}
