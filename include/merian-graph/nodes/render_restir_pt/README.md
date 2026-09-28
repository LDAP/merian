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
path consumed. Light samples are one-sample mixtures of the scene's pool and cell techniques, taken
from the unjittered cell, so their density is a function of the point alone.

The integrand is measured in primary sample space everywhere but around the reconnection vertex
x_k, where both segments are measured in solid angle; the Jacobian of a shift is the ratio of the
geometry terms into x_k.

`shift/mapping`:

- `hybrid` replays the random numbers up to the first vertex whose previous lobe is rough
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
  and kept in proportion to the score (Junkins et al. 2026). The selection is not reciprocal, so
  every neighbor costs two shifts, as with `disk`.

The canonical path and its neighbors are resampled with the confidence-weighted defensive pairwise
MIS weights (Lin et al. 2022, eq. 38; Hedstrom et al. 2026, eq. 11).

## Temporal reuse

`temporal/reprojection` follows the terms of the papers: prior work *backprojects* a pixel into the
previous frame, reservoir splatting projects the previous primary hits forward.

- `backprojection` takes the reservoir of the pixel the motion vector points at and resamples it
  with the confidence-weighted balance heuristic.
- `subpixel backprojection, samples as-is` (area reservoirs) moves the pixel's footprint back by its
  motion and resamples the previous samples that fall into it, relabeled to it, with the confidence
  of the up to four pixels it overlaps (Zhang et al. 2024, 4.3.1, the fast option).
- `subpixel backprojection, guaranteed coverage` also moves the samples of those pixels that fall
  outside the footprint into it by one-pixel shifts, with MIS over the overlapped pixels (Zhang et
  al. 2024, 4.3.2, the robust option). It shifts every previous reservoir into its eight neighbors
  and costs about twice the time of the other modes. It diverges where geometry moves rigidly, such
  as a first-person weapon: the previous frame is traced in the current scene, and the samples it
  keeps carry integrands of that mismatch into the next frame.
- `splatting` pushes every previous reservoir onto the pixel its primary vertex projects to (Liu et
  al. 2025), and resamples all that landed pairwise. With area reservoirs a reservoir lands on the
  point of the film its vertex projects to, weighed by the Jacobian between the film areas of both
  frames, and every current path splats back to find the previous pixel its MIS weight is taken
  against.
- `splatting, backprojection fallback` backprojects where nothing landed. With area reservoirs it
  splats only: which of the two a pixel took would depend on where other samples landed.

`history cap` bounds the confidence of the history; `duplication cap` lowers it where the previous
frame's path repeats around the pixel (Lin et al. 2026, 6). `specular motion vectors` backproject
the specular share of a smooth surface by the virtual image it reflects, `disocclusion motion
vectors` follow the occluder out of a disocclusion (Zeng et al. 2021).

`dynamic scene update` re-traces a history path's suffix in the current scene, and evaluates a path
moved into the previous frame on that frame's vertex positions and transforms. Rays of the previous
frame are traced in the current scene, so the reuse stays biased where the visibility of an emitter
changes between frames, such as around animated flames.

## Not supported

The depth-of-field reconnection shift, a previous-frame acceleration structure, reservoir
compression, per-lobe BSDF evaluation, analytic lights and denoiser guide buffers.

## References

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
