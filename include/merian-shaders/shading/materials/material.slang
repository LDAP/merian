#pragma once

__exported import merian_shaders.utils.frame;
__exported import merian_shaders.utils.textures;
__exported import merian_shaders.utils.texture_manager;
__exported import merian_shaders.utils.random;
__exported import merian_shaders.shading.shading_data;
__exported import merian_shaders.shading.bsdfs.bsdf;

namespace merian {

[Flags]
enum MaterialFlags : uint8_t {
    NO_NORMAL_MAPPING,
}

// This represents the definition of the material (before being evaluated at a shading point), e.g.
// Disney, glTF.
//
// The get_sample() method evaluates the material maps (gloss, normal, ...) to create an instance of
// this material for the shading point.
interface MaterialModel {
    associatedtype MaterialSampleImpl : MaterialSample;

    MaterialSampleImpl get_sample(const ShadingData sd,
                                  const TextureManager texture_manager,
                                  const LODSampler lod_sampler);
}

// This represents the evaluated material at a point on a surface or in a volume — normals,
// textures, etc. are resolved here.
interface MaterialSample {

    associatedtype ShadingFunctionImpl : ShadingFunction;

    // The local shading frame (for surfaces).
    Frame get_shading_frame() {
        return IdentityFrame();
    }

    // The BSDF or PhaseFunction for this Material. BSDFs assume a local tangent space with n = (0,
    // 0, 1), t = (1, 0, 0), b = (0, 1, 0). The incident and outgoing direction point away from the
    // shading location.
    ShadingFunctionImpl get_shading_function();

    // ------------------------------------------

    // the (representative) shading normal in world space. I.e. to guide a denoiser.
    // Convenience for get_shading_frame().get_normal().
    float3 get_shading_normal() {
        return get_shading_frame().get_normal();
    }
}

}
