## Accumulation

Calculates accumulated irradiance and the first two moments.

- Includes a simple but effective firefly filter for real-time usage.
  Calculates percentiles within 8x8 clusters,
  then clamps the luminance by `upper_percentile + user_factor * (upper_percentile - lower_percentile)`

- Includes a simple but effective "adaptive alpha reduction".
  For history values that are larger than `upper_percentile` or lower than `lower_percentile` alpha can be reduced dynamically.
  This helps against smearing, ghosting and fireflies in the history.

The implementation of the "firefly filter" and "adaptive alpha reduction" follow
> Lucas Alber. (2024), Markov Chain Path Guiding for Real-Time Global Illumination and Single-Scattering, MSc Thesis, Karlsruhe Institute of Technology.

Inputs:

| Type       | Input name   | Description                                                                            | Delay |
|------------|--------------|----------------------------------------------------------------------------------------|-------|
| VkImageIn  | src          | irradiance in `rgb`, second moment `a`                                                 | no    |
| VkBufferIn | gbuffer      | GBuffer (see `gbuffer.slang`)                                                         | no    |
| VkImageIn  | mv           | motion vectors in `r` and `g` channel                                                  | no    |
| VkImageIn  | prev_out     | feedback last `out`                                                                  | 1     |
| VkImageIn  | prev_history | feedback last `history`                                                                | 1     |

Outputs:

| Type       | Output name   | Description                                                 | Format/Resolution           | Persistent |
|------------|---------------|-------------------------------------------------------------|-----------------------------|------------|
| VkImageOut | out           | exp average of irradiance in `rgb`, second moment in `a`    | user defined or like irr    | no         |
| VkImageOut | history       | raw encoded normal in `r`; f16 depth, f16 history in `g`    | R32G32Uint                  | no         |

`history` packs everything the node needs from the previous frame into one 8-byte texel:
the scattered reprojection reads touch half as many cache lines as gathering the previous
GBuffer and a separate history image (~14% faster at 1080p). Consumers read the history
length as `f16tof32(history.g >> 16)`.

Events:

- `clear`: Sent in `process` if the accumulation buffer is reset
