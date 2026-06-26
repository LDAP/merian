#include "merian-shaders/light-cache/hashed_irradiance_cache.hpp"

#include "merian/utils/properties.hpp"

namespace merian {

namespace {
constexpr const char* MODULE_PATH =
    "merian-shaders/light-cache/hashed-irradiance-cache.slang";
} // namespace

HashedIrradianceCache::HashedIrradianceCache(const ResourceAllocatorHandle& allocator,
                                             const uint32_t buffer_size,
                                             const uint32_t probe_count,
                                             const bool stochastic_interpolation)
    : allocator(allocator), buffer_size(buffer_size), probe_count(probe_count),
      stochastic_interpolation(stochastic_interpolation) {

    buffer = allocator->create_buffer(
        vk::DeviceSize(buffer_size) * sizeof(HashedIrradianceCacheVertex),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::NONE, "HashedIrradianceCache::buffer");

    composition = SlangComposition::create();
    composition->add_module_from_path(MODULE_PATH);
}

SlangCompositionHandle HashedIrradianceCache::query_device_support_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(MODULE_PATH);
    return composition;
}

void HashedIrradianceCache::reset(const CommandBufferHandle& cmd) {
    cmd->fill(buffer);
    cmd->barrier(buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite));
}

void HashedIrradianceCache::write_to(ShaderCursor cursor) const {
    cursor["grid"]["buffer"] = buffer;
    cursor["grid"]["grid_tan_alpha_half"] = grid_tan_alpha_half;
    cursor["grid"]["grid_min_width"] = grid_min_width;
    cursor["grid"]["grid_steps_per_unit_size"] = grid_steps_per_unit_size;
    cursor["grid"]["grid_power"] = grid_power;
}

void HashedIrradianceCache::properties(Properties& props) {
    props.config_float("grid tan(alpha/2)", grid_tan_alpha_half,
                       "Cache resolution, lower means higher resolution.", 0.0001F);
    props.config_float("grid steps per unit", grid_steps_per_unit_size, "", 0.1F);
    props.config_float("grid min width", grid_min_width, "", 0.001F);
    props.config_float("grid power", grid_power, "", 0.1F);
}

} // namespace merian
