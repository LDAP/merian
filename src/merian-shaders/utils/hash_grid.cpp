#include "merian-shaders/utils/hash_grid.hpp"

#include "merian/shader/slang_session.hpp"
#include "merian/utils/properties.hpp"

#include <fmt/format.h>

namespace merian {

namespace {

std::size_t reflect_stride(const ShaderCompileContextHandle& compile_context,
                           const SlangCompositionHandle& composition,
                           const std::string& element_type_name) {
    const std::string buffer_type_name =
        fmt::format("RWStructuredBuffer<{}, ScalarDataLayout>", element_type_name);
    const auto reflection =
        SlangSession::get_type_layout(compile_context, composition, buffer_type_name);
    auto* const element_type_layout = reflection.type_layout->getElementTypeLayout();
    const std::size_t stride = element_type_layout != nullptr
                                   ? element_type_layout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM)
                                   : 0;
    if (stride == 0) {
        throw ShaderCompiler::compilation_failed(
            fmt::format("failed to reflect element stride for {}", buffer_type_name));
    }
    return stride;
}

} // namespace

HashGrid::HashGrid(const ShaderCompileContextHandle& compile_context,
                   const ResourceAllocatorHandle& allocator,
                   const SlangCompositionHandle& composition,
                   const std::string& data_type_name,
                   const uint32_t buffer_size,
                   const bool split_storage)
    : buffer_size(buffer_size), split_storage(split_storage) {

    // In the combined layout keys stays as a dummy to keep its binding valid.
    constexpr vk::DeviceSize dummy_size = 16;
    const vk::DeviceSize slot_stride =
        reflect_stride(compile_context, composition,
                       fmt::format("merian::HashGridSlot<{}, {}>", data_type_name,
                                   split_storage ? "true" : "false"));

    const auto usage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    keys = allocator->create_buffer(
        split_storage ? vk::DeviceSize(buffer_size) * 2 * sizeof(uint32_t) : dummy_size, usage,
        MemoryMappingType::NONE, "HashGrid::keys");
    data = allocator->create_buffer(buffer_size * slot_stride, usage, MemoryMappingType::NONE,
                                    "HashGrid::data");
}

void HashGrid::reset(const CommandBufferHandle& cmd) {
    cmd->fill(keys);
    cmd->fill(data);
    const std::array<vk::BufferMemoryBarrier2, 2> barriers = {
        keys->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer,
                              vk::PipelineStageFlagBits2::eAllCommands,
                              vk::AccessFlagBits2::eTransferWrite,
                              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite),
        data->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer,
                              vk::PipelineStageFlagBits2::eAllCommands,
                              vk::AccessFlagBits2::eTransferWrite,
                              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite),
    };
    cmd->barrier(barriers);
}

void HashGrid::write_to(ShaderCursor cursor) const {
    // Specialization may eliminate the unused keys buffer from the parameter block.
    if (auto keys_cursor = cursor["keys"]; keys_cursor.is_valid()) {
        keys_cursor = keys;
    }
    cursor["data"] = data;
    cursor["grid_tan_alpha_half"] = grid_tan_alpha_half;
    cursor["grid_level_bias"] = grid_level_bias;
    cursor["grid_distribution_dimension"] = grid_distribution_dimension;
}

void HashGrid::properties(Properties& props) {
    props.config_float("grid tan(alpha/2)", grid_tan_alpha_half,
                       "Cache resolution, lower means higher resolution.", 0.0001F);
    props.config_float("grid level bias", grid_level_bias,
                       "SHARC-style LOD bias; shifts level quantization / near-camera detail "
                       "(0 = neutral, fractional values shift the phase).",
                       0.05F);
    props.config_float("grid distribution dimension", grid_distribution_dimension,
                       "Spatial dimensionality the distributed levels are spread over "
                       "(2 = surface, 3 = volume). Smaller widens the spread.",
                       0.01F);
}

} // namespace merian
