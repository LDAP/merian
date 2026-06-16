import merian_shaders.shading.materials.material_system;

struct Params {
    ParameterBlock<merian::MaterialSystem> ms;
    RWStructuredBuffer<uint> output;
    uint material_id;
};

[shader("compute")][numthreads(1, 1, 1)] void main(ParameterBlock<Params> params) {
    // Get the material via dynamic dispatch
    merian::MaterialModel mat = params.ms.get_material(params.material_id);

    // Create shading data for evaluation
    merian::ShadingData sd;
    sd.pos = float3(0, 0, 0);
    sd.uv = half2(0.5, 0.5);
    sd.frame = merian::OrthonormalFrame(float3(0, 0, 1));
    sd.face_normal = float3(0, 0, 1);

    // Get sample using LOD0 (works in compute, no implicit derivatives)
    merian::LOD0Sampler lod;
    // texture_manager is a ParameterBlock<TextureManager> nested inside MaterialSystem
    merian::MaterialSample sample = mat.get_sample(sd, params.ms.texture_manager, lod);

    // Evaluate the BSDF: wi and wo pointing straight up in local tangent space
    float3 wi = float3(0, 0, 1);
    float3 wo = float3(0, 0, 1);
    merian::RandomGenerator rng = merian::XorShift32(42);
    float3 val = sample.get_bsdf().eval(wi, wo, rng);

    // For a Lambert BRDF with albedo (r,g,b): eval = INV_PI * albedo * wo.z = INV_PI * albedo
    params.output[0] = asuint(val.r);
    params.output[1] = asuint(val.g);
    params.output[2] = asuint(val.b);
}
