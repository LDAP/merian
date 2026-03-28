struct CBData {
    float x;
    int y;
};

struct Inner {
    ConstantBuffer<CBData> cb;
    uint z;
    RWStructuredBuffer<uint> output;
};

struct Params {
    ParameterBlock<Inner> inner;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.inner.output[0] = asuint(params.inner.cb.x);
    params.inner.output[1] = asuint(params.inner.cb.y);
    params.inner.output[2] = params.inner.z;
}
