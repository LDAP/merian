#pragma once

__exported import merian_shaders.utils.texture_manager;

namespace merian {

struct MaterialData {}

struct MaterialHeader {

    uint alpha_texture_id();
}

struct MaterialSystem {
    uint material_count;
    StructuredBuffer<MaterialData> materials;

    ParameterBlock<TextureManager> texture_manager;

    // --------------------------------------

    // return true if transparent / should be discarded
    bool alpha_test(const uint material_id, const half2 uv) {}
}

}
