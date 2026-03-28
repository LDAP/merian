struct Params {
    RWStructuredBuffer<uint> input;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.output[0] = params.input[0] + 1;
    params.output[1] = params.input[1] * 2;
    params.output[2] = params.input[2];
}
