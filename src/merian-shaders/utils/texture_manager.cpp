#include "merian-shaders/utils/texture_manager.hpp"

#include "merian/vk/utils/blits.hpp"

#include <cassert>
#include <cmath>
#include <fmt/format.h>

namespace merian {

TextureManager::TextureManager(const ShaderCompileContextHandle& compile_context,
                               const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator,
                               const ShaderObjectAllocatorHandle& obj_allocator,
                               const uint32_t initial_capacity)
    : compile_context(compile_context), context(context), allocator(allocator),
      obj_allocator(obj_allocator) {
    textures.resize(initial_capacity);

    // Build composition once — subsequent changes modify in-place.
    composition = SlangComposition::create();
    composition->add_module_from_path("merian-shaders/utils/texture-manager.slang");
    update_composition_constants();

    layout_program = SlangProgram::create(compile_context, composition);
    rebuild_shader_object();
}

void TextureManager::update_composition_constants() {
    composition->add_module_from_string("texture_manager_constants",
                                        fmt::format("namespace merian {{ export static const int "
                                                    "merian_texture_manager_texture_count = {}; }}",
                                                    static_cast<uint32_t>(textures.size())));
}

void TextureManager::rebuild_shader_object() {
    shader_object =
        layout_program->create_shader_object(context, "merian::TextureManager", obj_allocator);

    // Reinitialize all texture slots
    const auto& dummy = allocator->get_dummy_texture();
    auto cursor = shader_object->get_cursor();
    for (uint32_t i = 0; i < textures.size(); i++) {
        cursor["textures"][i] = textures[i] ? textures[i] : dummy;
    }
}

void TextureManager::update(const CommandBufferHandle& cmd) {
    if (pending_uploads.empty()) {
        return;
    }

    // 1. Transition every pending image to TransferDstOptimal (batched).
    std::vector<vk::ImageMemoryBarrier2> to_transfer_dst;
    to_transfer_dst.reserve(pending_uploads.size());
    for (const auto& copy : pending_uploads) {
        to_transfer_dst.push_back(copy.dst->barrier2(vk::ImageLayout::eTransferDstOptimal, true));
    }
    cmd->barrier(to_transfer_dst);

    // 2. Record copies and generate mip chains.
    for (const auto& copy : pending_uploads) {
        cmd->copy(copy.src, copy.dst, copy.region);
        cmd_generate_mipmaps(cmd, copy.dst);
    }

    // 3. Transition every pending image to ShaderReadOnlyOptimal (batched).
    // Descriptors were written upfront with this layout already.
    std::vector<vk::ImageMemoryBarrier2> to_shader_read;
    to_shader_read.reserve(pending_uploads.size());
    for (const auto& copy : pending_uploads) {
        to_shader_read.push_back(copy.dst->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
    }
    cmd->barrier(to_shader_read);

    pending_uploads.clear();
}

void TextureManager::resize(const uint32_t capacity) {
    if (capacity == textures.size()) {
        return;
    }

    textures.resize(capacity);
    update_composition_constants();
    rebuild_shader_object();
    increment_version();
}

TextureID TextureManager::allocate_id() {
    const TextureID id = ids.acquire();
    if (id >= textures.size()) {
        resize(static_cast<uint32_t>(textures.size()) * 2);
    }
    return id;
}

TextureID TextureManager::add_texture(const TextureHandle& texture) {
    const TextureID id = allocate_id();
    set_texture(id, texture);
    return id;
}

void TextureManager::set_texture(const TextureID id, const TextureHandle& texture) {
    assert(id < textures.size());
    ids.acquire(id);
    textures[id] = texture;
    shader_object->get_cursor()["textures"][id] =
        texture ? texture : allocator->get_dummy_texture();
}

TextureID TextureManager::add_texture_from_rgba8(const CommandBufferHandle& cmd,
                                                 const uint32_t* data,
                                                 const uint32_t width,
                                                 const uint32_t height,
                                                 const vk::SamplerAddressMode address_mode,
                                                 const vk::Filter mag_filter,
                                                 const vk::Filter min_filter,
                                                 const bool srgb,
                                                 const bool generate_mipmaps) {
    const TextureID id = allocate_id();
    set_texture_from_rgba8(id, cmd, data, width, height, address_mode, mag_filter, min_filter, srgb,
                           generate_mipmaps);
    return id;
}

void TextureManager::set_texture_from_rgba8(const TextureID id,
                                            const CommandBufferHandle& cmd,
                                            const uint32_t* data,
                                            const uint32_t width,
                                            const uint32_t height,
                                            const vk::SamplerAddressMode address_mode,
                                            const vk::Filter mag_filter,
                                            const vk::Filter min_filter,
                                            const bool srgb,
                                            const bool generate_mipmaps) {
    auto texture = allocator->create_texture_from_rgba8(
        cmd, data, width, height, address_mode, mag_filter, min_filter, srgb, "", generate_mipmaps);
    cmd->barrier(texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
    set_texture(id, texture);
}

TextureID TextureManager::add_texture_from_rgba8(const uint32_t* data,
                                                 const uint32_t width,
                                                 const uint32_t height,
                                                 const vk::SamplerAddressMode address_mode,
                                                 const vk::Filter mag_filter,
                                                 const vk::Filter min_filter,
                                                 const bool srgb,
                                                 const bool generate_mipmaps) {
    const TextureID id = allocate_id();
    set_texture_from_rgba8(id, data, width, height, address_mode, mag_filter, min_filter, srgb,
                           generate_mipmaps);
    return id;
}

void TextureManager::set_texture_from_rgba8(const TextureID id,
                                            const uint32_t* data,
                                            const uint32_t width,
                                            const uint32_t height,
                                            const vk::SamplerAddressMode address_mode,
                                            const vk::Filter mag_filter,
                                            const vk::Filter min_filter,
                                            const bool srgb,
                                            const bool generate_mipmaps) {
    assert(id < textures.size());
    const TextureHandle texture = stage_rgba8(data, width, height, address_mode, mag_filter,
                                              min_filter, srgb, generate_mipmaps);
    ids.acquire(id);
    textures[id] = texture;
    // Image is in eUndefined now and will reach eShaderReadOnlyOptimal during the next update().
    // Declaring the access layout up front lets us write the descriptor immediately.
    shader_object->get_cursor()["textures"][id].write(texture,
                                                      vk::ImageLayout::eShaderReadOnlyOptimal);
}

void TextureManager::remove_texture(const TextureID id) {
    assert(id < textures.size());
    [[maybe_unused]] const bool was_used = ids.release(id);
    assert(was_used && "Removing already-removed texture");

    textures[id].reset();
    shader_object->get_cursor()["textures"][id] = allocator->get_dummy_texture();
}

TextureHandle TextureManager::stage_rgba8(const uint32_t* data,
                                          const uint32_t width,
                                          const uint32_t height,
                                          const vk::SamplerAddressMode address_mode,
                                          const vk::Filter mag_filter,
                                          const vk::Filter min_filter,
                                          const bool srgb,
                                          const bool generate_mipmaps) {
    const uint32_t mip_levels =
        generate_mipmaps ? static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))) + 1)
                         : 1u;

    vk::ImageUsageFlags usage =
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
    if (generate_mipmaps) {
        usage |= vk::ImageUsageFlagBits::eTransferSrc;
    }

    const vk::ImageCreateInfo info{
        {},
        vk::ImageType::e2D,
        srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
        {width, height, 1},
        mip_levels,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        usage,
    };

    const ImageHandle image = allocator->create_image(info);
    pending_uploads.push_back(allocator->get_staging()->to_device(image, data));

    const SamplerHandle sampler = allocator->get_sampler_pool()->for_filter_and_address_mode(
        mag_filter, min_filter, address_mode);
    return allocator->create_texture(image, image->make_view_create_info(), sampler);
}

} // namespace merian
