__exported import merian_shaders.utils.frame;
__exported import merian_shaders.utils.textures;
__exported import merian_shaders.utils.texture_manager;
__exported import merian_shaders.utils.random;
__exported import merian_shaders.shading.shading_data;
__exported import merian_shaders.shading.bsdfs.bsdf;
__exported import merian_shaders.shading.materials.material_system_data;

namespace merian {

[Flags]
enum MaterialFlags : uint8_t {
    NO_NORMAL_MAPPING,
}

// Hint for material models to clamp their shading normal against the geometric normal,
// avoiding normal-map reflection artifacts (Keller et al. [2017]). Override via the
// composition (MaterialSystem::set_clamp_normals()).
extern static const bool merian_hint_clamp_normals = true;

// Lower bound on perceptual roughness, avoiding the degenerate zero-roughness GGX lobe.
extern static const float merian_hint_min_roughness = 0.0316f;

// This represents the definition of the material (before being evaluated at a shading point), e.g.
// Disney, glTF.
//
// The get_sample() method evaluates the material maps (gloss, normal, ...) to create an instance of
// this material for the shading point.
//
// Materials must ALWAYS have MaterialHeader as first member variable. The CPU side material system
// checks for this requirement.
interface MaterialModel {
    associatedtype MaterialSampleImpl : MaterialSample;

    MaterialSampleImpl get_sample(const ShadingData sd, const TextureManager texture_manager,
                                  const LODSampler2D lod_sampler);
}

// This represents the evaluated material at a point on a surface or in a volume — normals,
// textures, etc. are resolved here.
interface MaterialSample {

    associatedtype BSDFImpl : BSDF;

    property float3 emission {
        get;
    }

    // The local shading frame (for surfaces).
    property OrthonormalFrame shading_frame {
        get;
    }

    // The BSDF for this Material. BSDFs assume a local tangent space with n = (0,
    // 0, 1), t = (1, 0, 0), b = (0, 1, 0). The incident and outgoing direction point away from the
    // shading location.
    BSDFImpl get_bsdf();

    // ------------------------------------------

    // the (representative) shading normal in world space. I.e. to guide a denoiser.
    property float3 shading_normal {
        get;
    }

    // Relative IOR (incident / transmitted) at this surface; 1.0 for opaque materials. The
    // integrator multiplies the medium IOR by 1/eta after a transmission event.
    float get_eta() {
        return 1.0f;
    }

    // Volume absorption coefficient (per channel) of the medium entered on a transmission event;
    // 0 outside / for opaque materials. The integrator applies Beer-Lambert exp(-sigma * distance).
    float3 get_absorption() {
        return float3(0.0f);
    }
}

} // namespace merian
