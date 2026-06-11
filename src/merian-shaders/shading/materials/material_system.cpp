#include "merian-shaders/shading/materials/material_system.hpp"

#include "merian/utils/properties.hpp"

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
                               const TextureManagerHandle& texture_manager)
    : compile_context(compile_context), context(context), allocator(allocator),
      texture_manager(texture_manager) {

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
    shader_object = Versioned<ShaderObject>([this] { return build_shader_object(); });
    shader_object.depends_on(layout_program);
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
    composition->add_module_from_string("material_system_alpha_threshold",
                                        fmt::format("namespace merian {{ export static const float "
                                                    "merian_alpha_test_threshold = {:.6f}; }}",
                                                    threshold));
}

void MaterialSystem::set_clamp_normals(const bool clamp) {
    if (clamp == clamp_normals) {
        return;
    }
    clamp_normals = clamp;
    composition->add_module_from_string("material_system_clamp_normals",
                                        fmt::format("namespace merian {{ export static const bool "
                                                    "merian_hint_clamp_normals = {}; }}",
                                                    clamp ? "true" : "false"));
}

void MaterialSystem::set_min_roughness(const float min_roughness) {
    if (min_roughness == this->min_roughness) {
        return;
    }
    this->min_roughness = min_roughness;
    composition->add_module_from_string("material_system_min_roughness",
                                        fmt::format("namespace merian {{ export static const float "
                                                    "merian_hint_min_roughness = {:.9f}; }}",
                                                    min_roughness));
}

void MaterialSystem::set_enable_transmission(const bool enable) {
    enable_transmission = enable;
    composition->add_module_from_string("material_system_enable_transmission",
                                        fmt::format("namespace merian {{ export static const bool "
                                                    "merian_hint_enable_transmission = {}; }}",
                                                    enable ? "true" : "false"));
}

void MaterialSystem::set_enable_volume(const bool enable) {
    enable_volume = enable;
    composition->add_module_from_string("material_system_enable_volume",
                                        fmt::format("namespace merian {{ export static const bool "
                                                    "merian_hint_enable_volume = {}; }}",
                                                    enable ? "true" : "false"));
}

void MaterialSystem::set_enable_clearcoat(const bool enable) {
    enable_clearcoat = enable;
    composition->add_module_from_string("material_system_enable_clearcoat",
                                        fmt::format("namespace merian {{ export static const bool "
                                                    "merian_hint_enable_clearcoat = {}; }}",
                                                    enable ? "true" : "false"));
}

void MaterialSystem::set_enable_sheen(const bool enable) {
    enable_sheen = enable;
    composition->add_module_from_string("material_system_enable_sheen",
                                        fmt::format("namespace merian {{ export static const bool "
                                                    "merian_hint_enable_sheen = {}; }}",
                                                    enable ? "true" : "false"));
}

void MaterialSystem::set_enable_iridescence(const bool enable) {
    enable_iridescence = enable;
    composition->add_module_from_string("material_system_enable_iridescence",
                                        fmt::format("namespace merian {{ export static const bool "
                                                    "merian_hint_enable_iridescence = {}; }}",
                                                    enable ? "true" : "false"));
}

void MaterialSystem::properties(Properties& props) {
    float alpha = alpha_test_threshold;
    if (props.config_float("Alpha Test Threshold", alpha, "", 0.01F)) {
        set_alpha_test_threshold(alpha);
    }

    bool clamp = clamp_normals;
    if (props.config_bool("Clamp Normals", clamp,
                          "Hint the materials to clamp their normals to prevent artifacts when "
                          "using normal maps")) {
        set_clamp_normals(clamp);
    }

    float roughness = min_roughness;
    if (props.config_float(
            "Min Roughness", roughness,
            "Lower bound on roughness, preventing the degenerate zero-roughness lobe", 1e-3F, 0.0F,
            1.0F)) {
        set_min_roughness(roughness);
    }

    bool transmission = enable_transmission;
    if (props.config_bool("Enable Transmission", transmission,
                          "Render refraction through glass and other dielectrics; transmissive "
                          "materials shade as opaque when off")) {
        set_enable_transmission(transmission);
    }

    bool volume = enable_volume;
    if (props.config_bool("Enable Volume Absorption", volume,
                          "Tint light by the distance it travels through transmissive media, for "
                          "coloured glass and liquids")) {
        set_enable_volume(volume);
    }

    bool clearcoat = enable_clearcoat;
    if (props.config_bool("Enable Clearcoat", clearcoat,
                          "Add a thin glossy dielectric coat over the surface, for car paint, "
                          "lacquer and similar")) {
        set_enable_clearcoat(clearcoat);
    }

    bool sheen = enable_sheen;
    if (props.config_bool("Enable Sheen", sheen,
                          "Add a retroreflective microfibre sheen layer over the surface, for "
                          "cloth, velvet and fabric")) {
        set_enable_sheen(sheen);
    }

    bool iridescence = enable_iridescence;
    if (props.config_bool("Enable Iridescence", iridescence,
                          "Add thin-film interference to the specular reflection, for soap "
                          "bubbles, oil films and similar")) {
        set_enable_iridescence(iridescence);
    }
}

ShaderObjectHandle MaterialSystem::build_shader_object() const {
    SPDLOG_DEBUG("recreate shader object");

    const ShaderObjectHandle object =
        layout_program.get()->create_shader_object(context, "merian::MaterialSystem", allocator);
    auto cursor = object->get_cursor();
    cursor["texture_manager"] = texture_manager->get_shader_object();
    if (material_buffer) {
        cursor["material_count"] = static_cast<uint32_t>(materials.size());
        cursor["materials"] = material_buffer;
    }
    return object;
}

MaterialModelID MaterialSystem::register_material_type(const std::string& slang_type_name,
                                                       const std::string& slang_module_path) {
    if (auto it = material_types.find(slang_type_name); it != material_types.end()) {
        assert(it->second.slang_module_path == slang_module_path);
        return it->second.dispatch_id;
    }

    const auto dispatch_id = static_cast<MaterialModelID>(material_types.size());
    material_types.emplace(slang_type_name, MaterialTypeInfo{slang_module_path, dispatch_id});

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
        // Update constants in-place — bumps the composition version, so the program recompiles
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
    // Reset the feature toggles so a previous scene's lobes don't stay compiled in.
    set_enable_transmission(false);
    set_enable_volume(false);
    set_enable_clearcoat(false);
    // Keep material_buffer handle and max_payload_size; the next add_material
    // call will repopulate from index 0.
}

void MaterialSystem::update(const CommandBufferHandle& cmd) {
    texture_manager->update(cmd);

    // current even on frames with no material upload (e.g. a texture-manager resize only)
    shader_object.get();

    if (dirty_begin >= dirty_end || materials.empty())
        return;

    const vk::DeviceSize total_size = host_buffer.size();

    if (!material_buffer || material_buffer->get_size() < total_size) {
        cmd->keep_until_pool_reset(material_buffer);
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

    auto cursor = shader_object.get()->get_cursor();
    cursor["material_count"] = static_cast<uint32_t>(materials.size());
    cursor["materials"] = material_buffer;

    dirty_begin = UINT32_MAX;
    dirty_end = 0;
}

} // namespace merian
