__exported import merian_shaders.utils.frame;
__exported import merian_shaders.utils.textures;
__exported import merian_shaders.utils.texture_manager;
__exported import merian_shaders.utils.random;
__exported import merian_shaders.shading.shading_data;
__exported import merian_shaders.shading.homogeneous_volume;
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
                                  const LODSampler2D lod_sampler, out float3 out_emission,
                                  out OrthonormalFrame out_shading_frame);

    float3 get_emission(const ShadingData sd, const TextureManager texture_manager,
                        const LODSampler2D lod_sampler) {
        float3 emission;
        OrthonormalFrame shading_frame;
        get_sample(sd, texture_manager, lod_sampler, emission, shading_frame);
        return emission;
    }
}

// This represents how the evaluated material at a point on a surface or in a volume scatters.
interface MaterialSample {

    associatedtype BSDFImpl : BSDF;

    // The BSDF for this Material. BSDFs assume a local tangent space with n = (0,
    // 0, 1), t = (1, 0, 0), b = (0, 1, 0). The incident and outgoing direction point away from the
    // shading location.
    BSDFImpl get_bsdf();

    // ------------------------------------------

    // The medium on the far side of a transmission event, including the relative IOR of the
    // boundary. Opaque materials leave it at vacuum; the integrator returns to the enclosing
    // (exterior) medium when the ray leaves the surface.
    IsotropicVolume get_interior_volume() {
        return IsotropicVolume(float3(0.0f), float3(0.0f), 1.0f);
    }
}

} // namespace merian
