__exported import merian_shaders.utils.texture_manager_data;

namespace merian {

extern static const int merian_texture_manager_texture_count;

struct TextureManager {
    Sampler2D<float4> textures[merian_texture_manager_texture_count];

    // -----------------------------------------

    Sampler2D get_sampler2D(const TextureID texture_id) {
        return textures[NonUniformResourceIndex(texture_id)];
    }
}

}
