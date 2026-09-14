__exported import merian_shaders.gbuffer_data;
__exported import merian_gbuffer_fields;

namespace merian {

interface IWGBuffer {
    uint2 get_dimensions();

    void store(uint2 pixel, GBufferSample sample);
}

// Fields the connected gbuffer does not hold read as NaN (an invalid Hit).
extern struct GBuffer : IGBuffer;
extern struct WGBuffer : IWGBuffer;

extension GBuffer {
    SurfaceData get_surface(uint2 pixel) {
        let texels = load(pixel);
        SurfaceData surface;
        surface.normal = texels.normal;
        surface.linear_z = texels.linear_z;
        surface.grad_z = texels.grad_z;
        surface.delta_z = texels.delta_z;
        return surface;
    }

    // Floored albedo for demodulation. Divider and multiplier must use the same value or
    // channels below the floor do not cancel; the floor bounds the amplification of the
    // demodulated signal.
    float3 get_demodulation_albedo(uint2 pixel) {
        return max(get_albedo(pixel), 1e-2f);
    }

    float2 get_motion_vectors_dilated(int2 pixel, int radius) {
        const int2 size = int2(get_dimensions());
        float2 longest = float2(0);
        for (int j = -radius; j <= radius; j++) {
            for (int i = -radius; i <= radius; i++) {
                const float2 v =
                    get_motion_vectors(uint2(clamp(pixel + int2(i, j), int2(0), size - 1)));
                if (dot(v, v) > dot(longest, longest))
                    longest = v;
            }
        }
        return longest;
    }
}

}
