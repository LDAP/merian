# ReSTIR PT with compatibility-guided neighbor selection

ReSTIR path tracing after Lin et al. (2022), with the spatial neighbors chosen by geometric
compatibility instead of uniformly over the disk (Junkins et al. 2026). Paths reconnect at their
second vertex; a shift keeps everything from there on and rebuilds the segment from the
destination's own primary vertex.

Every domain is a GBuffer pixel, so a shift reads the destination's primary hit instead of
retracing a primary ray, and the camera integrates neither a pixel filter nor a lens. That matches
the `Render (Path-traced)` node, which is what the output is meant to converge against; a renderer
that samples the film would need the lens-vertex and primary-hit-reconnection shifts of Area
ReSTIR, which this node does not implement.

Not carried over from the reference implementation: motion blur, depth of field, and the clamped
and robust gather mechanisms.

## Temporal reprojection

`reprojection` picks how the previous frame's reservoirs reach a pixel.

`gather` pulls the one this pixel's motion vector points at. `splat` pushes every reservoir onto
the pixel its own primary vertex projects to (Liu et al. 2025), which reprojects the vertex rather
than trusting a screen-space vector, and `splat, gather fallback` pulls along the motion vector at
the pixels nothing landed on. A pixel holds `splat capacity` reservoirs; any past that are dropped.

Splatting sends any number of reservoirs to one pixel, so they share the defensive pairwise weights
the spatial pass uses rather than the two-way balance heuristic a gather needs. Every reservoir is
splatted, empty ones included: dropping one would take a technique out of those weights, not just a
sample.

The reference keeps the splatted reservoirs in a compacted list built from a prefix sum over the
per-pixel counts; a fixed number of slots per pixel needs one dispatch instead of four for the same
resampling.

## Convergence

The reuse chain is what limits how closely the output tracks the path tracer. ReSTIR does not
converge at a fixed reuse width and a fixed confidence cap; the cap is what bounds how far it
drifts (Lin et al. 2022, section 6.4). Measured against `Render (Path-traced)` (pbrt scenes at
their own film resolution, 1024 against 3000 iterations), the mean stays within 0.03 - 0.08 %, and
the offset is proportional to `history cap`: it is 0.008 % per unit of it and vanishes with the cap
at 1, with the reuse disabled, or with either reuse pass on its own.

Splatting widens that offset where it moves history across pixel borders, since the destination
then carries a confidence its own domain never earned: veach-ajar and staircase stay at 0.05 %,
while bathroom reaches 1.9 % at the default cap of 20 and falls back to 0.3 % at a cap of 1.

## References

Formulas in the shaders cite these by author and equation number.

- **Chao 1982.** A General Purpose Unequal Probability Sampling Plan. *Biometrika* 69(3).
  doi:10.1093/biomet/69.3.653
- **Efraimidis 2015.** Weighted Random Sampling over Data Streams. LNCS 9295.
  doi:10.1007/978-3-319-24024-4_12
- **Hedstrom et al. 2026.** Stochastic Pairwise MIS for Unbiased Large-Kernel Reuse in Real-Time.
  *CGF*. doi:10.1111/cgf.70391 - equation 11 is the confidence-weighted form of the defensive
  pairwise weights used for spatial reuse.
- **Junkins et al. 2026.** Compatibility-Guided Neighbor Selection for ReSTIR. *PACMCGIT* 9(4).
  doi:10.1145/3820024
- **Keller et al. 2016.** Path Space Filtering. SPMS 163. doi:10.1007/978-3-319-33507-0_21
- **Lehtinen et al. 2013.** Gradient-Domain Metropolis Light Transport. *TOG* 32(4).
  doi:10.1145/2461912.2461943
- **Lin et al. 2022.** Generalized Resampled Importance Sampling: Foundations of ReSTIR. *TOG*
  41(4). doi:10.1145/3528223.3530158
- **Liu et al. 2025.** Reservoir Splatting for Temporal Path Resampling and Motion Blur.
  *SIGGRAPH*. doi:10.1145/3721238.3730646
- **Roberts 2018.** The Unreasonable Effectiveness of Quasirandom Sequences.
- **Shirley and Chiu 1997.** A Low Distortion Map Between Disk and Square. *JGT* 2(3).
  doi:10.1080/10867651.1997.10487479
