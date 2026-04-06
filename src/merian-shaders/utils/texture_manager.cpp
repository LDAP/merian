#include "merian-shaders/utils/texture_manager.hpp"

#include <cassert>
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

    // Initialize all slots with the dummy texture
    const auto& dummy = allocator->get_dummy_texture();
    auto cursor = shader_object->get_cursor();
    for (uint32_t i = 0; i < initial_capacity; i++) {
        cursor["textures"][i] = dummy;
    }
}

void TextureManager::update_composition_constants() {
    composition->add_module_from_string(
        "texture_manager_constants",
        fmt::format(
            "namespace merian {{ export static const int merian_texture_manager_texture_count = {}; }}",
            static_cast<uint32_t>(textures.size())));
}

void TextureManager::rebuild_shader_object() {
    shader_object = layout_program->create_shader_object(context, "merian::TextureManager", obj_allocator);

    // Reinitialize all texture slots
    const auto& dummy = allocator->get_dummy_texture();
    auto cursor = shader_object->get_cursor();
    for (uint32_t i = 0; i < textures.size(); i++) {
        if (textures[i]) {
            cursor["textures"][i] = textures[i];
        } else {
            cursor["textures"][i] = dummy;
        }
    }
}

TextureID TextureManager::add_texture(const TextureHandle& texture) {
    TextureID id;
    if (!free_list.empty()) {
        id = free_list.back();
        free_list.pop_back();
    } else {
        id = static_cast<TextureID>(texture_count);
        if (id >= textures.size()) {
            textures.resize(textures.size() * 2);
            update_composition_constants();
            rebuild_shader_object();
            increment_version();
        }
    }

    textures[id] = texture;
    texture_count++;
    shader_object->get_cursor()["textures"][id] = texture;
    return id;
}

TextureID TextureManager::add_texture_from_rgba8(const CommandBufferHandle& cmd,
                                                  const uint32_t* data,
                                                  const uint32_t width,
                                                  const uint32_t height,
                                                  const vk::Filter filter,
                                                  const bool srgb) {
    auto texture = allocator->create_texture_from_rgba8(cmd, data, width, height, filter, filter,
                                                        srgb);
    cmd->barrier(texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
    return add_texture(texture);
}

void TextureManager::remove_texture(const TextureID id) {
    assert(id < textures.size());
    assert(textures[id] && "Removing already-removed texture");
    textures[id].reset();
    free_list.push_back(id);
    texture_count--;
    shader_object->get_cursor()["textures"][id] = allocator->get_dummy_texture();
}

const TextureHandle& TextureManager::get_texture(const TextureID id) const {
    assert(id < textures.size());
    return textures[id];
}

} // namespace merian
