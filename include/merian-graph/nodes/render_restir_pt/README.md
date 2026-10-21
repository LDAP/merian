# ReSTIR PT

ReSTIR path tracing (Lin et al. 2022) with the algorithms of ReSTIR PT Enhanced (Lin et al. 2026):
the hybrid shift with footprint-based reconnection, reciprocal neighbor selection, a history cap
that follows sample duplication, and the temporal update for dynamic scenes. Compatibility-guided
neighbor selection (Junkins et al. 2026) is available as an alternative to the reciprocal pairs.
Area reservoirs (Zhang et al. 2024) come with the temporal reuse of Area ReSTIR and of reservoir
splatting (Liu et al. 2025).

Formulas in the shaders cite the references below by author and equation number.

## Paths

A pixel traces `samples per pixel` paths and resamples their light-carrying subpaths into one
reservoir. Every vertex draws its scatter direction and its light sample from random streams of
its own, seeded by the path, so a replay in another domain reproduces them whatever the rest of the
path consumed. A light sample is resampled from the light cut's `draws` by its unshadowed
contribution and weighed against the scatter technique with the generalized balance heuristic
(Bitterli et al. 2020); the cell lookup jitters with the vertex's light stream, so the density of
the draws is a function of the point and the path's seed. Light samples are taken only on the side
of the surface the scatter technique reaches. The light cut learns from the light samples of
training paths, traced from a tenth of the pixels and never reused, and the next frame draws from
what it learned.

The integrand is measured in primary sample space everywhere but around the reconnection vertex
x_k, where both segments are measured in solid angle, and at a light sample drawn behind x_k, which
is measured on the light; the Jacobian of a shift is the ratio of the geometry terms into x_k. A
path keeps that light sample, so no other frame has to draw it again from a light cut that changed
since.

`shift/mapping`:

- `hybrid` replays the random numbers up to the first vertex whose previous BSDF is rough
  (`min roughness`) and whose segment is long on both sides compared to the pixel footprint
  (`min footprint`, Lin et al. 2026, eq. 5), and reconnects there. Both thresholds are jittered per
  path. Paths through specular chains are replayed whole.
- `reconnection` reconnects at the second vertex.

`area reservoirs` integrates over the pixel area and the lens (Zhang et al. 2024): a reservoir
carries the point of the pixel its primary ray passes, and a shift traces the primary ray through
the same point of the destination pixel. Its primary rays see the instances of the GBuffer's mask.

## Spatial reuse

`spatial/neighbor selection`:

- `reciprocal`: pairs from a periodic tile, moved by a random translation, flip and transpose every
  round (Lin et al. 2026, 5.1). Both sides of a pair share their two shifts, so a neighbor costs
  one shift.
- `disk`: low-discrepancy offsets on a disk of `radius` pixels.
- `CGNS`: `cgns/candidates` pixels from the disk, scored by the compatibility of their primary hits
  and kept in proportion to the score (Junkins et al. 2026). The score already weighs normal and
  distance, so `geometry rejection` does not apply. The selection is not reciprocal, so every
  neighbor costs two shifts, as with `disk`.

The canonical path and its neighbors are resampled with the confidence-weighted defensive pairwise
MIS weights (Lin et al. 2022, eq. 38; Hedstrom et al. 2026, eq. 11).

## Temporal reuse

`temporal/reprojection` follows the terms of the papers: prior work *backprojects* a pixel into the
previous frame, reservoir splatting projects the previous primary hits forward.

- `backprojection` takes the reservoir of the pixel the motion vector points at and resamples it
  with the confidence-weighted balance heuristic.
- `subpixel backprojection, samples as-is` (area reservoirs) moves the pixel's footprint back by its
  motion and resamples the previous samples that fall into it, relabeled to it (Zhang et al. 2024,
  4.3.1, the fast option). A path is weighed against the confidence of the previous pixel it falls
  in, the only one that can have sampled it, and none where the footprint leaves the image.
- `subpixel backprojection, guaranteed coverage` also moves the samples of those pixels that fall
  outside the footprint into it by one-pixel shifts, with MIS over the overlapped pixels (Zhang et
  al. 2024, 4.3.2, the robust option). Where the footprint leaves the image no sample moves in, so
  paths there are not weighed against the history. It shifts every previous reservoir into its own
  pixel and its eight neighbors, re-tracing their suffixes in the previous frame like the pixel's
  path moved there, and costs about twice the time of the other modes.
- `splatting` pushes every previous reservoir onto the pixel its primary vertex projects to (Liu et
  al. 2025), and resamples those that landed pairwise. A pixel keeps a random subset of up to five
  of them, the same whatever order they arrive in. With area reservoirs a reservoir lands on the
  point of the film its vertex projects to, weighed by the Jacobian between the film areas of both
  frames and by the inverse of its chance to be kept, and every current path splats back to find
  the previous pixel its MIS weight is taken against.
- `splatting, backprojection fallback` backprojects where nothing landed. With area reservoirs it
  splats only: which of the two a pixel took would depend on where other samples landed.

`history cap` bounds the confidence of the history; `duplication cap` lowers it where the previous
frame's path repeats around the pixel (Lin et al. 2026, 6). `specular motion vectors` backproject
the specular share of a smooth surface by the virtual image it reflects, `disocclusion motion
vectors` follow the occluder out of a disocclusion (Zeng et al. 2021).

A path moved into the previous frame is evaluated on that frame's vertex positions, transforms,
materials, time, acceleration structures and light sampling, so the target of a history path in its
own domain is the one it stored. Texture slots that are reused and meshes that switch materials
between frames are not tracked.

## Biasing

The settings that remove a bias at a cost:

- `independent light feedback`: the light cut learns from training paths of their own instead of
  the light samples of the paths the reservoirs reuse.
- `re-trace suffixes`: the suffix of a temporal path is re-traced every frame in the frame it is
  evaluated in, instead of keeping what arrived at its reconnection vertex when it was sampled.
  Turning it off is biased unless the scene, its lights and their sampling stay fixed.

## Path debugger

Connected to the Path Debugger, `debugger records` writes either the `candidates` of the initial
pass, with their per-vertex densities and light samples, or the `shaded paths`: the path each
pixel shades, replayed in its own domain.

## Not supported

The depth-of-field reconnection shift, reservoir compression, per-lobe BSDF evaluation, analytic
lights and denoiser guide buffers.

## References

- **Bitterli et al. 2020.** Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with
  Dynamic Direct Lighting. *TOG* 39(4). doi:10.1145/3386569.3392481
- **Chao 1982.** A General Purpose Unequal Probability Sampling Plan. *Biometrika* 69(3).
  doi:10.1093/biomet/69.3.653
- **Efraimidis 2015.** Weighted Random Sampling over Data Streams. LNCS 9295.
  doi:10.1007/978-3-319-24024-4_12
- **Hedstrom et al. 2026.** Stochastic Pairwise MIS for Unbiased Large-Kernel Reuse in Real-Time.
  *CGF*. doi:10.1111/cgf.70391
- **Junkins et al. 2026.** Compatibility-Guided Neighbor Selection for ReSTIR. *PACMCGIT* 9(4).
  doi:10.1145/3820024
- **Keller et al. 2016.** Path Space Filtering. SPMS 163. doi:10.1007/978-3-319-33507-0_21
- **Lin et al. 2022.** Generalized Resampled Importance Sampling: Foundations of ReSTIR. *TOG*
  41(4). doi:10.1145/3528223.3530158
- **Lin et al. 2026.** ReSTIR PT Enhanced. Released with Falcor 9.
- **Liu et al. 2025.** Reservoir Splatting for Temporal Path Resampling and Motion Blur.
  *SIGGRAPH*. doi:10.1145/3721238.3730646
- **Roberts 2018.** The Unreasonable Effectiveness of Quasirandom Sequences.
- **Shirley and Chiu 1997.** A Low Distortion Map Between Disk and Square. *JGT* 2(3).
  doi:10.1080/10867651.1997.10487479
- **Zeng et al. 2021.** Temporally Reliable Motion Vectors for Real-time Ray Tracing. *CGF* 40(2).
- **Zhang et al. 2024.** Area ReSTIR: Resampling for Real-Time Defocus and Antialiasing. *TOG*
  43(4).
