struct Data {
    float x;
    int y;
};

ConstantBuffer<Data> cb;
RWStructuredBuffer<uint> output;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    output[0] = asuint(cb.x);
    output[1] = asuint(cb.y);
}
