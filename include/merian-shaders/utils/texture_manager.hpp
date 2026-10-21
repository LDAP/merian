#pragma once

#include "merian-shaders/utils/texture-manager-data.slangh"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/free_list.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <vector>

namespace merian {

class Properties;

class TextureManager : public std::enable_shared_from_this<TextureManager> {
  public:
    TextureManager(const ShaderCompileContextHandle& compile_context,
                   const ContextHandle& context,
                   const ResourceAllocatorHandle& allocator,
                   uint32_t initial_capacity = 4096);

    static SlangCompositionHandle query_device_support_composition();

    // the textures of the last update become the previous frame's
    void update(const CommandBufferHandle& cmd);

    // records the staged uploads without changing the bindings
    void upload(const CommandBufferHandle& cmd);

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

    // staged; copied by the next update() or upload()
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

    // staged; copied by the next update() or upload()
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

    bool has_texture(const TextureID id) const {
        return ids.is_used(id);
    }

    const TextureHandle& get_texture(TextureID id) const {
        assert(id < textures.size());
        return textures[id];
    }

    // Renewed whenever a texture is set or removed at id.
    uint64_t get_texture_serial(TextureID id) const {
        assert(id < serials.size());
        return serials[id];
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

    // Bumps whenever the composition changes.
    uint64_t version() const {
        return composition->version();
    }

    const ShaderObjectHandle& get_shader_object() const {
        return shader_object.get();
    }

    operator const ShaderObjectHandle&() const {
        return shader_object.get();
    }

    void properties(Properties& props);

  private:
    void update_composition_constants();
    ShaderObjectHandle build_shader_object() const;
    void write(const ShaderObjectHandle& object, uint32_t slot, const TextureHandle& texture) const;

    TextureID allocate_id();
    void mark_changed(TextureID id);
    // the upload becomes visible to shaders after the next update()
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
    std::vector<uint64_t> serials;
    uint64_t last_serial = 0;
    // last_serial at the last update
    uint64_t update_serial = 0;
    FreeList<TextureID> ids;
    SlangCompositionHandle composition;
    Versioned<SlangProgram> layout_program;
    Versioned<ShaderObject> shader_object;

    // as bound for this and for the previous frame
    std::vector<TextureHandle> bound;
    std::vector<TextureHandle> prev_bound;
    // ids changed since the last update and at the last update
    std::vector<TextureID> changed;
    std::vector<TextureID> last_changed;

    std::vector<StagingMemoryManager::DeviceImageCopy> pending_uploads;
};

using TextureManagerHandle = std::shared_ptr<TextureManager>;

} // namespace merian
