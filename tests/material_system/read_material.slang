import merian_shaders.shading.materials.material_system;

struct Params {
    ParameterBlock<merian::MaterialSystem> ms;
    RWStructuredBuffer<uint> output;
    uint material_id;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    merian::MaterialBufferEntry entry = params.ms.materials[params.material_id];
    // Write header fields
    params.output[0] = entry.header.material_model_type_id;
    params.output[1] = entry.header.alpha_texture_id;
    // Write payload (base_color_factor for DiffuseMaterial = 4 floats as uint)
    params.output[2] = entry.blob.blob[0];
    params.output[3] = entry.blob.blob[1];
    params.output[4] = entry.blob.blob[2];
    params.output[5] = entry.blob.blob[3];
    // material_count
    params.output[6] = params.ms.material_count;
}
