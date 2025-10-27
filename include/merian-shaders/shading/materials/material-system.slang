#pragma once

__exported import merian_shaders.utils.texture_manager;
__exported import merian_shaders.shading.materials.material;

#include "merian-shaders/shading/materials/material-system-data.slangh"

namespace merian {

extern static const int merian_material_system_payload_max_size;
// alpha values less than the threshold are considered transparent
static const float merian_alpha_test_threshold = 0.7;

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

    ParameterBlock<TextureManager> texture_manager;

    // --------------------------------------

    // return true if transparent / should be discarded
    bool alpha_test(const MaterialID material_id,
                    const float2 uv,
                    const LODSampler sampler = ImplicitLODSampler()) {
        const MaterialHeader header = materials[material_id].header;
        if (let texture_id = header.get_alpha_texture_id()) {
            const float alpha = sampler.sample(texture_manager.get_sampler2D(texture_id), uv);
            return alpha < merian_alpha_test_threshold;
        }

        return false;
    }

    MaterialModel get_material(const MaterialID material_id) {
        const uint material_model_type_id = materials[material_id].header.material_model_type_id;
        // the material_model_type_id must be registerd on CPU side as type conformance. The CPU
        // side MaterialSystem ensures that.
        return createDynamicObject<MaterialModel>(material_model_type_id, materials[material_id]);
    }

    MaterialSample get_material_sample(const MaterialID material_id,
                                       const ShadingData sd,
                                       const LODSampler lod_sampler) {
        const MaterialModel material_model = get_material(material_id);
        return material_model.get_sample(sd, texture_manager, lod_sampler);
    }
}

}
