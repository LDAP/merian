# Firefly filter

A simple but effective firefly filter for real-time usage.

Calculates percentiles within 8x8 clusters,
then clamps the luminance by `upper_percentile + user_factor * (upper_percentile - lower_percentile)`


Inputs:

| Type   | Input ID | Input name   | Description                                                 | Delay |
|--------|----------|--------------|-------------------------------------------------------------|-------|
| Image  | 1        | irr          | noisy irradiance                                            | no    |
| Image  | 2        | moments      | first and second moment in `rg`                             | no    |

Outputs:

| Type  | Output ID | Output name | Description                                                 | Format/Resolution        | Persistent |
|-------|-----------|-------------|-------------------------------------------------------------|--------------------------|------------|
| Image | 0         | irr         | clamped irradiance                                          | like input               | no         |
| Image | 1         | moments     | clamp-corrected moments                                     | like input               | no         |
