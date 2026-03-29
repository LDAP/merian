RWStructuredBuffer<uint> g_output;

struct Params {
    float x;
    int y;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    g_output[0] = asuint(params.x);
    g_output[1] = asuint(params.y);
}
