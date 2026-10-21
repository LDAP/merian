import merian_shaders.shading.materials.material_system;

struct Params {
    ParameterBlock<merian::MaterialSystem> ms;
    RWStructuredBuffer<uint> output;
    uint material_id;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    for (uint i = 0; i < 4; i++) {
        params.output[i] = params.ms.materials[params.material_id].blob.blob[i];
        params.output[4 + i] = params.ms.prev_materials[params.material_id].blob.blob[i];
    }
}
