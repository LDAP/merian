#include "merian-shaders/shading/materials/material_system.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fmt/format.h>

namespace merian {

static uint32_t payload_size_in_uints(uint32_t byte_size) {
    return (byte_size + 3) / 4;
}

MaterialSystem::MaterialSystem(const ShaderCompileContextHandle& compile_context,
                               const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator,
                               const ShaderObjectAllocatorHandle& obj_allocator,
                               const TextureManagerHandle& texture_manager)
    : compile_context(compile_context), context(context), allocator(allocator),
      obj_allocator(obj_allocator), texture_manager(texture_manager) {

    assert(context->get_device()
                   ->get_enabled_features()
                   .get_16_bit_storage_features()
                   .storageBuffer16BitAccess == VK_TRUE &&
           "MaterialSystem requires storageBuffer16BitAccess (enable the merian extension)");

    // Build the composition once — subsequent changes modify it in-place.
    composition = SlangComposition::create();
    composition->add_composition(texture_manager->get_composition());
    composition->add_module_from_path("merian-shaders/shading/materials/material-system.slang");
    update_composition_constants();

    layout_program = SlangProgram::create(compile_context, composition);
    layout_program->on_changed(layout_program, [&] { rebuild_shader_object(); });

    rebuild_shader_object();
}

void MaterialSystem::update_composition_constants() {
    const uint32_t payload_uints = std::max(payload_size_in_uints(max_payload_size), 1u);
    composition->add_module_from_string(
        "material_system_constants", fmt::format("namespace merian {{ export static const int "
                                                 "merian_material_system_payload_max_size = {}; }}",
                                                 payload_uints));
}

void MaterialSystem::set_alpha_test_threshold(const float threshold) {
    if (threshold == alpha_test_threshold) {
        return;
    }
    alpha_test_threshold = threshold;
    composition->add_module_from_string(
        "material_system_alpha_threshold",
        fmt::format("namespace merian {{ export static const float "
                    "merian_alpha_test_threshold = {:.6f}; }}",
                    threshold));
}

void MaterialSystem::rebuild_shader_object() {
    SPDLOG_DEBUG("recreate shader object");

    shader_object =
        layout_program->create_shader_object(context, "merian::MaterialSystem", obj_allocator);
    auto cursor = shader_object->get_cursor();
    cursor["texture_manager"] = texture_manager->get_shader_object();
    if (material_buffer) {
        cursor["material_count"] = static_cast<uint32_t>(materials.size());
        cursor["materials"] = material_buffer;
    }

    texture_manager->on_changed(shader_object, [&] {
        shader_object->get_cursor()["texture_manager"] = texture_manager->get_shader_object();
    });

    increment_version();
}

MaterialModelID MaterialSystem::register_material_type(const std::string& slang_type_name,
                                                       const std::string& slang_module_path) {
    auto dispatch_id = static_cast<MaterialModelID>(material_types.size());
    material_types.push_back({slang_type_name, slang_module_path, dispatch_id});

    // triggers program rebuild via listener
    composition->add_module_from_path(slang_module_path);
    composition->add_type_conformance("merian::MaterialModel", slang_type_name, dispatch_id);
    return dispatch_id;
}

uint32_t MaterialSystem::get_entry_size() const {
    return sizeof(MaterialHeader) + payload_size_in_uints(max_payload_size) * sizeof(uint32_t);
}

void MaterialSystem::repack_host_buffer() {
    const uint32_t entry_sz = get_entry_size();
    host_buffer.resize(entry_sz * materials.size(), 0);
    for (uint32_t i = 0; i < materials.size(); i++) {
        uint8_t* entry = host_buffer.data() + i * entry_sz;
        std::memcpy(entry, &materials[i].header, sizeof(MaterialHeader));
        std::memcpy(entry + sizeof(MaterialHeader), materials[i].payload.data(),
                    materials[i].payload.size());
    }
}

MaterialID MaterialSystem::add_material(const MaterialModelID type_id, const Material& material) {
    assert(type_id < material_types.size());

    StoredMaterial stored;
    stored.header = material.header;
    stored.header.material_model_type_id = type_id;

    uint32_t payload_bytes = material.get_payload_size();
    stored.payload.resize(payload_bytes);
    material.write_payload(stored.payload.data());

    if (payload_bytes > max_payload_size) {
        max_payload_size = payload_bytes;
        // Update constants in-place — triggers program rebuild via Versionable chain
        update_composition_constants();

        // Entry size changed — repack everything
        materials.push_back(std::move(stored));
        repack_host_buffer();
        dirty_begin = 0;
        dirty_end = static_cast<uint32_t>(materials.size());
        return static_cast<MaterialID>(materials.size() - 1);
    }

    auto id = static_cast<MaterialID>(materials.size());
    materials.push_back(std::move(stored));

    // Append to host buffer
    const uint32_t entry_sz = get_entry_size();
    host_buffer.resize(entry_sz * materials.size(), 0);
    uint8_t* entry = host_buffer.data() + id * entry_sz;
    std::memcpy(entry, &materials.back().header, sizeof(MaterialHeader));
    std::memcpy(entry + sizeof(MaterialHeader), materials.back().payload.data(),
                materials.back().payload.size());

    if (dirty_begin > id)
        dirty_begin = id;
    dirty_end = static_cast<uint32_t>(materials.size());

    return id;
}

void MaterialSystem::update_material(const MaterialID id, const Material& material) {
    assert(id < materials.size());

    const uint32_t payload_bytes = material.get_payload_size();
    assert(
        payload_bytes <= max_payload_size &&
        "update_material payload larger than current max_payload_size — repack not supported here");

    StoredMaterial& stored = materials[id];
    // Header may carry alpha_texture_id and similar fields; refresh in case the
    // caller updated them. Keep the dispatch type id we assigned at add time.
    const auto type_id = stored.header.material_model_type_id;
    stored.header = material.header;
    stored.header.material_model_type_id = type_id;

    stored.payload.resize(payload_bytes);
    material.write_payload(stored.payload.data());

    const uint32_t entry_sz = get_entry_size();
    uint8_t* entry = host_buffer.data() + static_cast<size_t>(id * entry_sz);
    std::memcpy(entry, &stored.header, sizeof(MaterialHeader));
    std::memcpy(entry + sizeof(MaterialHeader), stored.payload.data(), stored.payload.size());

    dirty_begin = std::min<uint32_t>(dirty_begin, id);
    const uint32_t end_after = id + 1u;
    dirty_end = std::max(dirty_end, end_after);
}

void MaterialSystem::clear() {
    materials.clear();
    host_buffer.clear();
    dirty_begin = UINT32_MAX;
    dirty_end = 0;
    // Keep material_buffer handle and max_payload_size; the next add_material
    // call will repopulate from index 0.
}

void MaterialSystem::update(const CommandBufferHandle& cmd) {
    texture_manager->update(cmd);

    if (dirty_begin >= dirty_end || materials.empty())
        return;

    const vk::DeviceSize total_size = host_buffer.size();

    if (!material_buffer || material_buffer->get_size() < total_size) {
        material_buffer = allocator->create_buffer(
            total_size,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::NONE, "MaterialSystem::material_buffer");
        // Buffer recreated — must upload everything
        dirty_begin = 0;
        dirty_end = static_cast<uint32_t>(materials.size());
    }

    const uint32_t entry_sz = get_entry_size();
    const vk::DeviceSize offset = dirty_begin * entry_sz;
    const vk::DeviceSize size = (dirty_end - dirty_begin) * entry_sz;

    allocator->get_staging()->cmd_to_device(cmd, material_buffer, host_buffer.data() + offset,
                                            offset, size);

    cmd->barrier(material_buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eShaderRead));

    auto cursor = shader_object->get_cursor();
    cursor["material_count"] = static_cast<uint32_t>(materials.size());
    cursor["materials"] = material_buffer;

    dirty_begin = UINT32_MAX;
    dirty_end = 0;
}

} // namespace merian
