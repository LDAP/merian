#pragma once

#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

class GBuffer {
  public:
    GBuffer(const ShaderCompileContextHandle& compile_context,
            const ContextHandle& context,
            const ResourceAllocatorHandle& allocator,
            const vk::Extent3D extent)
        : extent(extent) {
        SlangCompositionHandle composition = SlangComposition::create();
        composition->add_module_from_path("merian-shaders/gbuffer.slang");
        layout_program = SlangProgram::create(compile_context, composition);

        r_shader_object =
            layout_program->create_shader_object(context, "merian::GBuffer", allocator);
        w_shader_object =
            layout_program->create_shader_object(context, "merian::WGBuffer", allocator);
    }

    const ShaderObjectHandle& get_shader_object() const {
        return r_shader_object;
    }

    const ShaderObjectHandle& get_write_shader_object() const {
        return w_shader_object;
    }

    operator const ShaderObjectHandle&() const {
        return r_shader_object;
    }

    void set_resources(const ImageViewHandle& tex0,
                       const ImageViewHandle& tex1,
                       const ImageViewHandle& tex2,
                       const ImageViewHandle& tex3) {
        auto r_cursor = r_shader_object->get_cursor();
        r_cursor["tex0"] = tex0;
        r_cursor["tex1"] = tex1;
        r_cursor["tex2"] = tex2;
        r_cursor["tex3"] = tex3;
        auto w_cursor = w_shader_object->get_cursor();
        w_cursor["tex0"] = tex0;
        w_cursor["tex1"] = tex1;
        w_cursor["tex2"] = tex2;
        w_cursor["tex3"] = tex3;
    }

    vk::Extent3D get_extent() const {
        return extent;
    }

  private:
    const vk::Extent3D extent;

    SlangProgramHandle layout_program;

    ShaderObjectHandle r_shader_object;
    ShaderObjectHandle w_shader_object;
};

using GBufferHandle = std::shared_ptr<GBuffer>;

} // namespace merian
