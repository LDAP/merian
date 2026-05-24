#pragma once

#include "merian-shaders/utils/texture-manager-data.slangh"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/free_list.hpp"
#include "merian/utils/versionable.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <vector>

namespace merian {

class TextureManager : public Versionable, public std::enable_shared_from_this<TextureManager> {
  public:
    TextureManager(const ShaderCompileContextHandle& compile_context,
                   const ContextHandle& context,
                   const ResourceAllocatorHandle& allocator,
                   uint32_t initial_capacity = 4096);

    // Records pending uploads queued by the cmd-less add_/set_ overloads.
    void update(const CommandBufferHandle& cmd);

    void resize(const uint32_t capacity);

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

    // Stages the upload; the actual GPU copy is recorded by the next update().
    TextureID
    add_texture_from_rgba8(const uint32_t* data,
                           uint32_t width,
                           uint32_t height,
                           vk::SamplerAddressMode address_mode = vk::SamplerAddressMode::eRepeat,
                           vk::Filter mag_filter = vk::Filter::eLinear,
                           vk::Filter min_filter = vk::Filter::eLinear,
                           bool srgb = true,
                           bool generate_mipmaps = false);

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

    // Stages the upload; the actual GPU copy is recorded by the next update().
    void
    set_texture_from_rgba8(TextureID id,
                           const uint32_t* data,
                           uint32_t width,
                           uint32_t height,
                           vk::SamplerAddressMode address_mode = vk::SamplerAddressMode::eRepeat,
                           vk::Filter mag_filter = vk::Filter::eLinear,
                           vk::Filter min_filter = vk::Filter::eLinear,
                           bool srgb = true,
                           bool generate_mipmaps = false);

    void remove_texture(TextureID id);

    const TextureHandle& get_texture(TextureID id) const {
        assert(id < textures.size());
        return textures[id];
    }

    uint32_t get_texture_count() const {
        return ids.count();
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

    // Pulls the next slot from free_list or grows the table.
    TextureID allocate_id();
    // Builds image + view + sampler and queues the staging copy. Returns the
    // texture; the upload becomes visible to shaders after the next update().
    TextureHandle stage_rgba8(const uint32_t* data,
                              uint32_t width,
                              uint32_t height,
                              vk::SamplerAddressMode address_mode,
                              vk::Filter mag_filter,
                              vk::Filter min_filter,
                              bool srgb,
                              bool generate_mipmaps);

    ShaderCompileContextHandle compile_context;
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    std::vector<TextureHandle> textures;
    FreeList<TextureID> ids;
    SlangCompositionHandle composition;
    SlangProgramHandle layout_program;
    ShaderObjectHandle shader_object;

    std::vector<StagingMemoryManager::DeviceImageCopy> pending_uploads;
};

using TextureManagerHandle = std::shared_ptr<TextureManager>;

} // namespace merian
