#include "merian-shaders/light-cache/hashed_irradiance_cache.hpp"

#include "merian/utils/properties.hpp"

#include <fmt/format.h>

namespace merian {

namespace {
constexpr const char* MODULE_PATH =
    "merian-shaders/light-cache/hashed-irradiance-cache.slang";
} // namespace

HashedIrradianceCache::HashedIrradianceCache(const ShaderCompileContextHandle& compile_context,
                                             const ContextHandle& context,
                                             const ResourceAllocatorHandle& allocator,
                                             const uint32_t capacity)
    : compile_context(compile_context), context(context), allocator(allocator), capacity(capacity) {

    composition = SlangComposition::create();
    composition->add_module_from_path(MODULE_PATH);
    update_composition_constants();

    layout_program = SlangProgram::create(compile_context, composition);
    shader_object = Versioned<ShaderObject>([this] { return build_shader_object(); });
    shader_object.depends_on(layout_program);
}

SlangCompositionHandle HashedIrradianceCache::query_device_support_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(MODULE_PATH);
    return composition;
}

void HashedIrradianceCache::update_composition_constants() {
    composition->add_module_from_string(
        "hashed_irradiance_cache_constants",
        fmt::format("namespace merian {{\n"
                    "export static const float lc_grid_tan_alpha_half = {:f};\n"
                    "export static const float lc_grid_steps_per_unit_size = {:f};\n"
                    "export static const float lc_grid_min_width = {:f};\n"
                    "export static const float lc_grid_power = {:f};\n"
                    "}}",
                    lc_grid_tan_alpha_half, lc_grid_steps_per_unit_size, lc_grid_min_width,
                    lc_grid_power));
}

ShaderObjectHandle HashedIrradianceCache::build_shader_object() const {
    const ShaderObjectHandle object = layout_program->create_shader_object_for_type(
        context, "merian::HashedIrradianceCache", allocator);
    if (buffer) {
        object->get_cursor()["irr_cache"] = buffer;
    }
    return object;
}

void HashedIrradianceCache::update(const CommandBufferHandle& cmd) {
    const bool realloc = needs_realloc || !buffer;
    if (realloc) {
        if (buffer) {
            cmd->keep_until_pool_reset(buffer);
        }
        buffer = allocator->create_buffer(
            vk::DeviceSize(capacity) * sizeof(HashedIrradianceCacheVertex),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::NONE, "HashedIrradianceCache::buffer");
        needs_realloc = false;
        reset(cmd);
    }

    // Rebuilds if the composition (constants) changed; build_shader_object re-binds the buffer.
    const ShaderObjectHandle object = shader_object.get();
    if (realloc) {
        object->get_cursor()["irr_cache"] = buffer;
    }
}

void HashedIrradianceCache::reset(const CommandBufferHandle& cmd) {
    if (!buffer) {
        return;
    }
    cmd->fill(buffer);
    cmd->barrier(buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite));
}

void HashedIrradianceCache::properties(Properties& props) {
    if (props.config_uint("buffer capacity", capacity, "Number of cache slots backing the hash grid.")) {
        needs_realloc = true;
    }

    bool constants_changed = false;
    constants_changed |= props.config_float("grid tan(alpha/2)", lc_grid_tan_alpha_half,
                                            "Cache resolution, lower means higher resolution.",
                                            0.0001F);
    constants_changed |=
        props.config_float("grid steps per unit", lc_grid_steps_per_unit_size, "", 0.1F);
    constants_changed |= props.config_float("grid min width", lc_grid_min_width, "", 0.001F);
    constants_changed |= props.config_float("grid power", lc_grid_power, "", 0.1F);

    if (constants_changed) {
        update_composition_constants();
    }
}

} // namespace merian
