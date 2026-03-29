struct Params {
    float x;
    int y;
    RWStructuredBuffer<uint> output;
};

ParameterBlock<Params> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    params.output[0] = asuint(params.x);
    params.output[1] = asuint(params.y);
}
