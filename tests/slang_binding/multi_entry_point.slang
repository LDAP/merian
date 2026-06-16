// Two entry points, each with its own ParameterBlock. Compiled as one module (getTargetCode),
// Slang numbers descriptor-set register spaces globally across both entry points, so the second
// entry point's ParameterBlock does NOT land at space 0.
struct ParamsA {
    float x;
    RWStructuredBuffer<uint> output;
};

struct ParamsB {
    float y;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void mainA(ParameterBlock<ParamsA> a) {
    a.output[0] = asuint(a.x);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mainB(ParameterBlock<ParamsB> b) {
    b.output[0] = asuint(b.y);
}
