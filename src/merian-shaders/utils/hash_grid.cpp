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

    // The inactive layout's buffers stay as dummies to keep their bindings valid.
    constexpr vk::DeviceSize dummy_size = 16;
    const vk::DeviceSize data_stride =
        split_storage ? reflect_stride(compile_context, composition, data_type_name) : 0;
    const vk::DeviceSize record_stride =
        split_storage ? 0
                      : reflect_stride(compile_context, composition,
                                       fmt::format("merian::HashGridRecord<{}>", data_type_name));

    const auto usage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    keys = allocator->create_buffer(
        split_storage ? vk::DeviceSize(buffer_size) * 2 * sizeof(uint32_t) : dummy_size, usage,
        MemoryMappingType::NONE, "HashGrid::keys");
    data = allocator->create_buffer(split_storage ? vk::DeviceSize(buffer_size) * data_stride
                                                  : dummy_size,
                                    usage, MemoryMappingType::NONE, "HashGrid::data");
    records = allocator->create_buffer(split_storage ? dummy_size
                                                     : vk::DeviceSize(buffer_size) * record_stride,
                                       usage, MemoryMappingType::NONE, "HashGrid::records");
}

void HashGrid::reset(const CommandBufferHandle& cmd) {
    cmd->fill(keys);
    cmd->fill(data);
    cmd->fill(records);
    const std::array<vk::BufferMemoryBarrier2, 3> barriers = {
        keys->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer,
                              vk::PipelineStageFlagBits2::eAllCommands,
                              vk::AccessFlagBits2::eTransferWrite,
                              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite),
        data->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer,
                              vk::PipelineStageFlagBits2::eAllCommands,
                              vk::AccessFlagBits2::eTransferWrite,
                              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite),
        records->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite),
    };
    cmd->barrier(barriers);
}

void HashGrid::write_to(ShaderCursor cursor) const {
    // Specialization may eliminate the inactive layout's buffers from the parameter block.
    if (auto keys_cursor = cursor["keys"]; keys_cursor.is_valid()) {
        keys_cursor = keys;
    }
    if (auto data_cursor = cursor["data"]; data_cursor.is_valid()) {
        data_cursor = data;
    }
    if (auto records_cursor = cursor["records"]; records_cursor.is_valid()) {
        records_cursor = records;
    }
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
