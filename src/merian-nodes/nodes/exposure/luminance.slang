#include "layout.slangh"

import merian_shaders.utils.camera;
import merian_shaders.utils.textures;

using merian;

groupshared uint[workgroup_size_x * workgroup_size_y] local_hist;
groupshared uint count;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int local_index: SV_GroupIndex) {
    const uint hist_size = workgroup_size_x * workgroup_size_y;

    float max_luminance;
    if (params.automatic == 0) {
        const float ev = ev_100(params.aperature, params.shutter_time, params.iso);
        max_luminance = ev_100_to_max_luminance(ev, params.iso, params.q);
    } else {
        const uint local_bin = histogram[local_index];
        local_hist[local_index] = local_bin * local_index;
        if (local_index == 0)
            count = histogram[hist_size];
        GroupMemoryBarrierWithGroupSync();

        // sum all local_hist entries into local_hist[0]
        for (uint split = (hist_size >> 1); split > 0; split >>= 1) {
            if (uint(local_index) < split) {
                local_hist[local_index] += local_hist[local_index + split];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (local_index == 0) {
            const int2 image_size = texture_dimensions(img_src);
            const float num_bright_pixels = max(count - float(local_bin), 1.0);
            const float average_bin = (local_hist[0] / num_bright_pixels) - 1.0;
            const float average_l =
                exp2(((average_bin / float(hist_size - 2.)) * params.max_log_histogram) +
                     params.min_log_histogram);

            const float ev = ev_100_from_average(average_l, 100, params.K);
            max_luminance = ev_100_to_max_luminance(ev, 100, params.q);

            if (params.reset != 1) {
                float last_max_luminance = luminance[0];
                if (isnan(last_max_luminance) || isinf(last_max_luminance))
                    last_max_luminance = max_luminance;

                const float tau =
                    -params.timediff *
                    (max_luminance > last_max_luminance ? params.speed_up : params.speed_down);
                max_luminance =
                    last_max_luminance + (max_luminance - last_max_luminance) * (1 - exp(tau));
            }
        }
    }

    if (local_index == 0) {
        luminance[0] = max_luminance;
    }
}
