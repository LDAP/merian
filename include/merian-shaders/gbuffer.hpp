#pragma once

#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_program.hpp"
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
        const SlangProgramHandle program = SlangProgram::create(compile_context, composition).get();

        r_shader_object =
            program->create_shader_object_for_type(context, "merian::GBuffer", allocator);
        w_shader_object =
            program->create_shader_object_for_type(context, "merian::WGBuffer", allocator);
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
        for (const auto& object : {r_shader_object, w_shader_object}) {
            auto cursor = object->get_cursor();
            cursor["tex0"].write(tex0, vk::ImageLayout::eGeneral);
            cursor["tex1"].write(tex1, vk::ImageLayout::eGeneral);
            cursor["tex2"].write(tex2, vk::ImageLayout::eGeneral);
            cursor["tex3"].write(tex3, vk::ImageLayout::eGeneral);
        }
    }

    vk::Extent3D get_extent() const {
        return extent;
    }

  private:
    const vk::Extent3D extent;

    ShaderObjectHandle r_shader_object;
    ShaderObjectHandle w_shader_object;
};

using GBufferHandle = std::shared_ptr<GBuffer>;

} // namespace merian
