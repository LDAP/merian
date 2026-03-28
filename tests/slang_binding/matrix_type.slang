struct Params {
    float4x4 val_mat;
    RWStructuredBuffer<uint> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    // Write all 16 elements row by row
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            params.output[r * 4 + c] = asuint(params.val_mat[r][c]);
        }
    }
}
