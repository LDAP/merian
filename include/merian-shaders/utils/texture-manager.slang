__exported import merian_shaders.utils.texture_manager_data;
__exported import merian_shaders.utils.textures;

namespace merian {

extern static const int merian_texture_manager_slot_count;

struct TextureManager {
    // this frame's textures, then the previous frame's
    Sampler2D<float4> textures[merian_texture_manager_slot_count];

    Textures get_textures(const bool prev_frame = false) {
        return {this, prev_frame ? uint(merian_texture_manager_slot_count / 2) : 0u};
    }
}

// The textures of one frame.
struct Textures {
    TextureManager manager;
    uint base;

    Sampler2D<float4> get_sampler2D(const TextureID texture_id) {
        return manager.textures[NonUniformResourceIndex(base + texture_id)];
    }

    float4 sample(const LODSampler2D lod, const TextureID texture_id, const float2 uv) {
        return lod.sample(get_sampler2D(texture_id), uv);
    }
}

} // namespace merian
