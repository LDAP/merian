import merian_shaders.utils.textures;
__exported import merian_shaders.scene.hit;

namespace merian {

// f16 saturates above this
static const float MAX_LINEAR_Z = 65504.0;

static const float GBUFFER_UNREQUESTED = asfloat(0x7FC00000u);

// The longest motion vector in a (2 * radius + 1)^2 neighbourhood, so a reprojection follows the
// fastest thing covering the pixel rather than whatever the center happens to hit. Takes the
// vectors from the first two channels of any 2D texture, sampled or storage.
float2 dilate_motion_vectors<let N : int, int sampleCount, int isCombined, int format>(
    const _Texture<vector<float, N>, __Shape2D, 0, 0, sampleCount, 0, 0, isCombined, format> mv,
    const int2 pixel,
    const int radius) {
    const int2 tex_size = int2(texture_dimensions(mv));
    float2 longest = float2(0);
    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            const vector<float, N> texel =
                mv[uint2(clamp(pixel + int2(i, j), int2(0), tex_size - 1))];
            const float2 v = float2(texel[0], texel[1]);
            if (dot(v, v) > dot(longest, longest))
                longest = v;
        }
    }
    return longest;
}

struct PackedHitInfo {
    uint instance_id;
    uint primitive_id;
    uint16_t geometry_index;
    uint16_t instance_index;
    half2 barycentrics;

    static PackedHitInfo pack(Hit hit) {
        PackedHitInfo p;
        p.instance_id = hit.instance_id;
        p.primitive_id = hit.primitive_id;
        p.geometry_index = uint16_t(hit.geometry_index);
        p.instance_index = uint16_t(hit.instance_index);
        p.barycentrics = half2(hit.barycentrics);
        return p;
    }

    Hit unpack() {
        return Hit(instance_id, uint(instance_index), primitive_id, uint(geometry_index),
                   float2(barycentrics));
    }
}

struct SurfaceData {
    float3 normal;
    float linear_z;
    half2 grad_z;
    float delta_z;
};

}
