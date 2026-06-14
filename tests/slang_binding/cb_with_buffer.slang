// ConstantBuffer holding a resource descriptor (a buffer), used to exercise descriptor
// reassignment: swapping the ConstantBuffer must repoint the resource binding, leaving no
// reference to the replaced object's buffer.
struct CBData {
    float scale;
    RWStructuredBuffer<uint> data;
};

struct Params {
    ConstantBuffer<CBData> cb;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.output[0] = params.cb.data[0];
    params.output[1] = asuint(params.cb.scale);
}
