__exported import merian_shaders.shading.materials.material;

namespace merian {

extern static const int merian_material_system_payload_max_size;
// alpha values less than the threshold are considered transparent.
extern static const float merian_alpha_test_threshold = 0.5;
// False where no material can have anything but vacuum behind a transmission event.
extern static const bool merian_material_system_any_interior_volume = false;

struct MaterialPayload {
    uint blob[merian_material_system_payload_max_size];
}

// The opaque material buffer entry.
struct MaterialBufferEntry {
    MaterialHeader header;
    MaterialPayload blob;
}

struct MaterialSystem {
    uint material_count;
    StructuredBuffer<MaterialBufferEntry> materials;
    StructuredBuffer<MaterialBufferEntry> prev_materials;

    ParameterBlock<TextureManager> texture_manager;

    // true where the texel is discarded
    bool alpha_test(const MaterialID material_id,
                    const float2 uv,
                    const LODSampler2D lod_sampler = DefaultLODSampler(),
                    const bool prev_frame = false) {
        return get_alpha(material_id, uv, lod_sampler, prev_frame) < merian_alpha_test_threshold;
    }

    // Sampled alpha (or 1.0 if the material has no alpha texture).
    float get_alpha(const MaterialID material_id,
                    const float2 uv,
                    const LODSampler2D lod_sampler = DefaultLODSampler(),
                    const bool prev_frame = false) {
        const MaterialHeader header =
            prev_frame ? prev_materials[material_id].header : materials[material_id].header;
        if (let texture_id = header.get_alpha_texture_id()) {
            return texture_manager.get_textures(prev_frame).sample(lod_sampler, texture_id, uv).a;
        }
        return 1.f;
    }

    MaterialModel get_material(const MaterialID material_id, const bool prev_frame = false) {
        const MaterialBufferEntry entry =
            prev_frame ? prev_materials[material_id] : materials[material_id];
        return createDynamicObject<MaterialModel>(entry.header.material_model_type_id, entry);
    }

    float3 get_emission(const MaterialID material_id,
                        const ShadingData sd,
                        const LODSampler2D lod_sampler = DefaultLODSampler(),
                        const bool prev_frame = false) {
        return get_material(material_id, prev_frame)
            .get_emission(sd, texture_manager.get_textures(prev_frame), lod_sampler);
    }

    MaterialSample get_material_sample(const MaterialID material_id,
                                       const ShadingData sd,
                                       out float3 out_emission,
                                       out OrthonormalFrame out_shading_frame,
                                       const LODSampler2D lod_sampler = DefaultLODSampler(),
                                       const bool prev_frame = false) {
        const MaterialModel material_model = get_material(material_id, prev_frame);
        return material_model.get_sample(sd, texture_manager.get_textures(prev_frame), lod_sampler,
                                         out_emission, out_shading_frame);
    }
}

} // namespace merian
