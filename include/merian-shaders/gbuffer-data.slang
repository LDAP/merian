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
    uint channels[5];

    __init(const uint c0, const uint c1, const uint c2, const uint c3, const uint c4) {
        channels = {c0, c1, c2, c3, c4};
    }

    static PackedHitInfo pack(const Hit hit) {
        return PackedHitInfo(hit.instance_id, hit.primitive_id,
                             (hit.geometry_index & 0xFFFFu) | (hit.instance_index << 16),
                             asuint(hit.barycentrics.x), asuint(hit.barycentrics.y));
    }

    Hit unpack() {
        return Hit(channels[0], channels[2] >> 16, channels[1], channels[2] & 0xFFFFu,
                   asfloat(uint2(channels[3], channels[4])));
    }
}

struct SurfaceData {
    float3 normal;
    float linear_z;
    half2 grad_z;
    float delta_z;
};

}
