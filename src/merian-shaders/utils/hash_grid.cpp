#include "merian-shaders/utils/hash_grid.hpp"

#include "merian/shader/slang_session.hpp"
#include "merian/utils/properties.hpp"

#include <fmt/format.h>

namespace merian {

HashGrid::HashGrid(const ShaderCompileContextHandle& compile_context,
                   const ResourceAllocatorHandle& allocator,
                   const SlangCompositionHandle& composition,
                   const std::string& data_type_name,
                   const uint32_t buffer_size)
    : buffer_size(buffer_size) {

    const std::string buffer_type_name = fmt::format(
        "RWStructuredBuffer<merian::HashGridVertex<{}>, ScalarDataLayout>", data_type_name);
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

    buffer = allocator->create_buffer(vk::DeviceSize(buffer_size) * stride,
                                      vk::BufferUsageFlagBits::eStorageBuffer |
                                          vk::BufferUsageFlagBits::eTransferDst,
                                      MemoryMappingType::NONE, "HashGrid::buffer");
}

void HashGrid::reset(const CommandBufferHandle& cmd) {
    cmd->fill(buffer);
    cmd->barrier(buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite));
}

void HashGrid::write_to(ShaderCursor cursor) const {
    cursor["buffer"] = buffer;
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
