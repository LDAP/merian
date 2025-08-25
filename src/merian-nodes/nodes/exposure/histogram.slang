#include "layout.slangh"

import merian_shaders.colors.colorspaces;
import merian_shaders.utils.textures;

using merian;

groupshared uint[workgroup_size_x * workgroup_size_y] local_hist;
groupshared uint count;

uint get_bin(const float3 color, const uint hist_size) {
    float l = XYZ_luminance(color);
    if (l < pow(2, params.min_log_histogram)) {
        return 0;
    }

    const float normalized = (log2(l) - params.min_log_histogram) / params.max_log_histogram;
    const uint bin = uint(normalized * (hist_size - 2) + 1);
    return clamp(bin, 0, hist_size - 1);
}

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID, int local_index: SV_GroupIndex) {
    const uint hist_size = workgroup_size_x * workgroup_size_y;

    local_hist[local_index] = 0;
    if (local_index == 0)
        count = 0;
    GroupMemoryBarrierWithGroupSync();

    if (is_on_texture(ipos, img_out)) {
        bool mask;
        int2 tex_dims = texture_dimensions(img_out);
        if (params.metering == 0) {
            mask = true;
        } else if (params.metering == 1) {
            // Dirty but works :D
            mask = true;
            mask = mask && (ipos.x % max(uint(smoothstep(.1 * tex_dims.r / 2, tex_dims.r / 2,
                                                         distance(tex_dims / 2, ipos)) *
                                              13),
                                         1)) == 0;
            mask = mask && (ipos.y % max(uint(smoothstep(.1 * tex_dims.r / 2, tex_dims.r / 2,
                                                         distance(tex_dims / 2, ipos)) *
                                              7),
                                         1)) == 0;
        } else if (params.metering == 2) {
            mask = distance(tex_dims / 2, ipos) / min(tex_dims.r, tex_dims.g) < .5;
        } else {
            // should never happen
            mask = false;
        }

        if (mask) {
            const float3 color = img_src.Load(int3(ipos, 0)).rgb;
            const uint bin = get_bin(color, hist_size);
            InterlockedAdd(local_hist[bin], 1);
            InterlockedAdd(count, 1);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    InterlockedAdd(histogram[local_index], local_hist[local_index]);
    // Number of pixel that are represented in the histogram
    if (local_index == 0) {
        InterlockedAdd(histogram[hist_size], count);
    }
}
