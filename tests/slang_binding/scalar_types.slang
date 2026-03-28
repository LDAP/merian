struct Params {
    float val_float;
    int val_int;
    uint val_uint;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    params.output[0] = asuint(params.val_float);
    params.output[1] = asuint(params.val_int);
    params.output[2] = params.val_uint;
}
