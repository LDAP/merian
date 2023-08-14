# SVGF

Denoiser for real-time ray tracing.

Follows the implementation of:

- Schied, Kaplanyan, Chaitanya, Burgess, Dachsbacher, and Lefohn: *Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination*, HPG 2017.
- Christoph Schied, Christoph Peters, Carsten Dachsbacher: *Gradient Estimation for Real-Time Adaptive Temporal Filtering*

Inputs:

| Type  | Input ID | Input name   | Description                                                 | Delay |
|-------|----------|--------------|-------------------------------------------------------------|-------|
| Image | 0        | prev_out     | feedback last `out`                                         | 1     |
|       |
| Image | 1        | irr          | (accumulated) noisy irradiance, history length in `a`       | no    |
| Image | 2        | moments      | first and second moment in `rg`                             | no    |
|       |
| Image | 3        | gbuf         | GBuffer (enc normal in r, linear depth in g, monents in ba) | no    |
| Image | 4        | prev_gbuf    | previous GBuffer                                            | 1     |
|       |
| Image | 5        | albedo       | the demodulated albedo                                      | no    |
| Image | 6        | mv           | motion vectors in `r` and `g` channel                       | no    |

Outputs:

| Type  | Output ID | Output name | Description                                                 | Format/Resolution        | Persistent |
|-------|-----------|-------------|-------------------------------------------------------------|--------------------------|------------|
| Image | 0         | out         | the final image                                             | same as `irr`            | no         |


```
Filter and variance estimation is adapted from:

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

