## SVGF

Denoiser for real-time ray tracing.

Follows the implementation of:

- Schied, Kaplanyan, Chaitanya, Burgess, Dachsbacher, and Lefohn: *Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination*, HPG 2017.
- Christoph Schied, Christoph Peters, Carsten Dachsbacher: *Gradient Estimation for Real-Time Adaptive Temporal Filtering*

But modifies the original code in that:

- Spacial variance estimate is disabled (led to flickering)
- Uses shared memory in the variance estimate
- Slightly modified edge detecting functions
- Provides a bunch of debugging options

#### Inputs:

| Type   | Input name   | Description                                                 | Delay |
|--------|--------------|-------------------------------------------------------------|-------|
| Image  | prev_out     | feedback last `out`                                         | 1     |
|        |
| Image  | irr          | (accumulated) noisy irradiance, history length in `a`       | no    |
| Image  | moments      | first and second moment in `rg`                             | no    |
|        |
| Image  | albedo       | the demodulated albedo                                      | no    |
| Image  | mv           | motion vectors in `r` and `g` channel                       | no    |
|        |
| Buffer | gbuffer      | GBuffer (see `gbuffer.glsl.h`)                              | no    |
| Buffer | prev_gbuffer | previous GBuffer                                            | 1     |

#### Outputs:

| Type  | Output name | Description                                                 | Format/Resolution        | Persistent |
|-------|-------------|-------------------------------------------------------------|--------------------------|------------|
| Image | out         | the final image (containing remaining variance in a)        | same as `irr`            | no         |


#### Copyright notice

Filter and variance estimation is adapted from:

```
Copyright (c) 2018, Christoph Schied
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Karlsruhe Institute of Technology nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

