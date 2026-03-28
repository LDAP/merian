struct Level3 {
    float a;
};

struct Level2 {
    ConstantBuffer<Level3> level3;
    int b;
};

struct Level1 {
    ConstantBuffer<Level2> level2;
    uint c;
};

struct Params {
    ConstantBuffer<Level1> level1;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.output[0] = asuint(params.level1.level2.level3.a);
    params.output[1] = asuint(params.level1.level2.b);
    params.output[2] = params.level1.c;
}
