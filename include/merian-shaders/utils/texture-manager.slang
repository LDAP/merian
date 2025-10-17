#pragma once

namespace merian {

extern static const int merian_texture_manager_texture_count;

struct TextureManager {
    Sampler2D<float4> textures[merian_texture_manager_texture_count];
}

}
