## Shader Library (merian-shaders)

Merian ships a collection of reusable shader code under `include/merian-shaders/`. Slang is the preferred language; legacy GLSL files are also present.

### Using Slang modules

Import a module by path or name in your Slang shader:

```slang
import "merian-shaders/utils/random";
import "merian-shaders/utils/sampling";
import "merian-shaders/colors/tonemaps";
```

---

### colors/

Color space conversions and tone mapping.

| Module | Content |
|---|---|
| `colorspace_srgb.slang` | sRGB ↔ linear |
| `colorspace_oklab.slang` | OKLab perceptually uniform space |
| `colorspace_oklch.slang` | OKLCh cylindrical space |
| `colorspace_rec2020.slang` | BT.2020 / Rec. 2020 |
| `colorspace_xyz.slang` | CIE XYZ |
| `colorspace_yuv.slang` | YUV |
| `colorspace_dng.slang` | DNG / camera color space |
| `colorspace_dtucs.slang` | DT UCS uniform color space |
| `colorspace_munsell.slang` | Munsell color system |
| `colorspaces.slang` | Aggregate import of all color spaces |
| `tonemaps.slang` | Reinhard, ACES, AgX, and other tone mapping operators |

### distributions/

Statistical distributions for sampling.

| Module | Content |
|---|---|
| `distribution.slang` | Base distribution interface |
| `von-mises-fisher.slang` | Von Mises-Fisher distribution for directional sampling |

### scene/

Scene-level structures.

| Module | Content |
|---|---|
| `camera.slang` | Camera parameters, ray generation |
| `acceleration-structure.slang` | TLAS/BLAS query helpers |
| `environment-map.slang` | Environment map sampling |
| `scene.slang` | Composed scene structure (camera + environment + geometry) |

### shading/bsdfs/

Bidirectional scattering distribution functions.

| Module | Content |
|---|---|
| `bsdf.slang` | Base BSDF interface |
| `brdf-lambert-diffuse.slang` | Lambertian diffuse |
| `brdf-ggx.slang` | GGX specular (Cook-Torrance) |

### shading/materials/

Material system built on top of BSDFs.

| Module | Content |
|---|---|
| `material.slang` | Material definition interface |
| `material-system.slang` | Material system (dispatch by material type) |

### shading/phase-functions/

Volume scattering phase functions.

| Module | Content |
|---|---|
| `phase-function.slang` | Base phase function interface |
| `isotropic.slang` | Isotropic scattering |
| `rayleigh.slang` | Rayleigh scattering |
| `henyey-greenstein.slang` | Henyey-Greenstein |
| `draine.slang` | Draine scattering |
| `mie-approx.slang` | Approximate Mie scattering |

### shading/

High-level shading utilities.

| Module | Content |
|---|---|
| `shading-data.slang` | Surface and volumetric shading data structures |
| `shading-function.slang` | Evaluate shading given BSDF + light + shading data |

### utils/

General-purpose shader utilities.

| Module | Content |
|---|---|
| `random.slang` | PCG, Xorshift RNGs; tea hash |
| `sampling.slang` | Cosine, hemisphere, sphere, disk, triangle sampling |
| `math.slang` | Math helpers (orthonormal basis, saturate, ...) |
| `hash.slang` | Hash functions |
| `encoding.slang` | Normal map encode/decode, octahedral, packed formats |
| `camera.slang` | Camera utility functions |
| `envmap.slang` | Environment map utilities |
| `fresnel.slang` | Fresnel reflectance (Schlick, exact) |
| `frame.slang` | Shading frame / TBN operations |
| `grid.slang` | Grid-based operations |
| `interpolation.slang` | Bilinear, trilinear, smoothstep |
| `morton.slang` | Morton code (Z-order curve) |
| `bit-twiddling.slang` | Bit manipulation utilities |
| `ray-differential.slang` | Ray differentials for texture filtering in ray tracing |
| `raytrace.slang` | Ray tracing utilities and primitive intersections |
| `reprojection.slang` | Temporal reprojection helpers |
| `shared-memory.slang` | LDS / workgroup shared memory utilities |
| `texture-manager.slang` | Bindless texture management |
| `textures.slang` | Texture sampling and filtering |
| `volumes.slang` | Volumetric rendering utilities |

### Root-level

| Module | Content |
|---|---|
| `gbuffer.slang` | G-buffer layout and pack/unpack helpers |

---

### Legacy GLSL

GLSL equivalents (`*.glsl`) of most modules are present for backwards compatibility. New code should use the Slang modules.
