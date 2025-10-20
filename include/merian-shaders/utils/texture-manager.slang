#pragma once

namespace merian {

typealias TextureID = uint16_t;

extern static const int merian_texture_manager_texture_count;

struct TextureManager {
    Sampler2D<float4> textures[merian_texture_manager_texture_count];

    // -----------------------------------------

    Sampler2D get_sampler2D(const TextureID texture_id) {
        return textures[texture_id];
    }
}

}
