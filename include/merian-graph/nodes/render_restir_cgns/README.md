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

Not carried over from the reference implementation: reservoir splatting and the scatter-based
temporal modes, motion blur, depth of field, Russian roulette, and the clamped and robust gather
mechanisms. The default configuration uses none of them.

The reuse chain is what limits how closely the output tracks the path tracer. Every resampling
step assumes the reservoirs it combines are independent, which neighbouring pixels stop being once
they have shared history for a while. Measured against `Render (Path-traced)` (pbrt scenes, 1024
against 3000 iterations), the mean stays within 0.03 - 0.08 %, and the offset is proportional to
`history cap`: it is 0.008 % per unit of it and vanishes with the cap at 1, with the reuse
disabled, or with either reuse pass on its own.
