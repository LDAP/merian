#pragma once

#include "merian-shaders/utils/texture-manager-data.slangh"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/versionable.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <vector>

namespace merian {

class TextureManager : public Versionable,
                       public std::enable_shared_from_this<TextureManager> {
  public:
    TextureManager(const ShaderCompileContextHandle& compile_context,
                   const ContextHandle& context,
                   const ResourceAllocatorHandle& allocator,
                   const ShaderObjectAllocatorHandle& obj_allocator,
                   uint32_t initial_capacity = 4096);

    TextureID add_texture(const TextureHandle& texture);

    TextureID
    add_texture_from_rgba8(const CommandBufferHandle& cmd,
                           const uint32_t* data,
                           uint32_t width,
                           uint32_t height,
                           vk::SamplerAddressMode address_mode = vk::SamplerAddressMode::eRepeat,
                           vk::Filter mag_filter = vk::Filter::eLinear,
                           vk::Filter min_filter = vk::Filter::eLinear,
                           bool srgb = true,
                           bool generate_mipmaps = false);

    // Place a texture at a specific predefined slot. Asserts id < capacity().
    // Replaces any existing handle and updates the shader cursor. Use this
    // mode when the caller owns a stable external ID space (e.g. Quake's
    // gl_texnum). Mutually exclusive with add_texture per manager instance.
    void set_texture(TextureID id, const TextureHandle& texture);

    // Convenience: upload data and set_texture(id, ...).
    void
    set_texture_from_rgba8(TextureID id,
                           const CommandBufferHandle& cmd,
                           const uint32_t* data,
                           uint32_t width,
                           uint32_t height,
                           vk::SamplerAddressMode address_mode = vk::SamplerAddressMode::eRepeat,
                           vk::Filter mag_filter = vk::Filter::eLinear,
                           vk::Filter min_filter = vk::Filter::eLinear,
                           bool srgb = true,
                           bool generate_mipmaps = false);

    void remove_texture(TextureID id);

    const TextureHandle& get_texture(TextureID id) const;

    uint32_t get_texture_count() const {
        return texture_count;
    }

    uint32_t get_capacity() const {
        return static_cast<uint32_t>(textures.size());
    }

    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    const ShaderObjectHandle& get_shader_object() const {
        return shader_object;
    }

    operator const ShaderObjectHandle&() const {
        return shader_object;
    }

  private:
    void update_composition_constants();
    void rebuild_shader_object();

    ShaderCompileContextHandle compile_context;
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderObjectAllocatorHandle obj_allocator;
    std::vector<TextureHandle> textures;
    std::vector<TextureID> free_list;
    uint32_t texture_count = 0;
    SlangCompositionHandle composition;
    SlangProgramHandle layout_program;
    ShaderObjectHandle shader_object;
};

using TextureManagerHandle = std::shared_ptr<TextureManager>;

} // namespace merian
