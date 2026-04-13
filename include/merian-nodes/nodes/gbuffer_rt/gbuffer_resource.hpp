#pragma once

#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

// Host-side GBuffer resource passed between nodes via PtrOut/PtrIn.
// Holds the texture handles, shader object, and composition for link-time binding.
//
// Similar to Scene: consumers call get_composition() to link against the GBuffer module,
// and get_shader_object() to bind the textures at dispatch time.
class GBufferResource {
  public:
    GBufferResource(const TextureHandle& tex0,
                    const TextureHandle& tex1,
                    const TextureHandle& tex2,
                    const ShaderObjectHandle& shader_object,
                    const SlangCompositionHandle& composition)
        : tex0(tex0), tex1(tex1), tex2(tex2),
          shader_object(shader_object), composition(composition) {}

    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    const ShaderObjectHandle& get_shader_object() const {
        return shader_object;
    }

    operator const ShaderObjectHandle&() const {
        return shader_object;
    }

    const TextureHandle& get_tex0() const {
        return tex0;
    }

    const TextureHandle& get_tex1() const {
        return tex1;
    }

    const TextureHandle& get_tex2() const {
        return tex2;
    }

    vk::Extent3D get_extent() const {
        return tex0->get_image()->get_extent();
    }

  private:
    TextureHandle tex0;
    TextureHandle tex1;
    TextureHandle tex2;
    ShaderObjectHandle shader_object;
    SlangCompositionHandle composition;
};

using GBufferResourceHandle = std::shared_ptr<GBufferResource>;

} // namespace merian
