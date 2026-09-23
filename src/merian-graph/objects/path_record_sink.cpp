#include "merian-graph/objects/path_record_sink.hpp"

#include <algorithm>
#include <array>

namespace merian {

OutputConnectorDescriptor PathRecordSink::describe_output(const ContextHandle& context,
                                                          const vk::Extent3D& extent,
                                                          const uint32_t spp,
                                                          const uint32_t vertices_per_path) {
    last_max_range = context->get_physical_device()->get_device_limits().maxStorageBufferRange &
                     ~vk::DeviceSize{3};
    last_extent = extent;
    vk::DeviceSize size = vk::DeviceSize{PATH_RECORD_HEADER_UINTS} * 4;
    if (is_connected) {
        size = stream_size(spp, vertices_per_path);
    }
    emitted_size = size;
    con = ManagedVkBufferOut::create(vk::BufferCreateInfo(
        {}, size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc));
    return {"path_records", con,
            ConnectorAccess::ray_tracing_read_write | ConnectorAccess::compute_read_write};
}

bool PathRecordSink::update_connected(const NodeIOLayout& io_layout) {
    if (io_layout.is_connected(con) == is_connected) {
        return false;
    }
    is_connected = !is_connected;
    return true;
}

void PathRecordSink::add_constants(const SlangCompositionHandle& composition) const {
    composition->add_module_from_string(
        "path_record_constants",
        is_connected ? "import merian_shaders.debug.path_record;\n"
                       "namespace merian {\n"
                       "export static const bool merian_path_record = true;\n"
                       "export struct PathRecorder : IPathRecorder = LivePathRecorder;\n"
                       "}"
                     : "import merian_shaders.debug.path_record;\n"
                       "namespace merian {\n"
                       "export static const bool merian_path_record = false;\n"
                       "export struct PathRecorder : IPathRecorder = NullPathRecorder;\n"
                       "}");
}

void PathRecordSink::process(const CommandBufferHandle& cmd,
                             const NodeIO& io,
                             const uint64_t iteration,
                             ShaderCursor cursor) const {
    if (!is_connected) {
        return;
    }
    // graph memory is uninitialized, and a garbage header would arm recording
    if (iteration == 0) {
        const std::array<const uint32_t, PATH_RECORD_HEADER_UINTS> zero{};
        const BufferHandle& buffer = io[con];
        cmd->update(buffer, vk::ArrayProxy<const uint32_t>(zero));
        cmd->barrier(buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer,
                                             vk::PipelineStageFlagBits2::eRayTracingShaderKHR |
                                                 vk::PipelineStageFlagBits2::eComputeShader,
                                             vk::AccessFlagBits2::eTransferWrite,
                                             vk::AccessFlagBits2::eShaderRead |
                                                 vk::AccessFlagBits2::eShaderWrite));
    }
    bind(io, cursor);
}

void PathRecordSink::bind(const NodeIO& io, ShaderCursor cursor) const {
    if (is_connected && cursor.is_valid()) {
        cursor = static_cast<const BufferHandle&>(io[con]);
    }
}

bool PathRecordSink::properties(Properties& config,
                                const uint32_t spp,
                                const uint32_t vertices_per_path) {
    if (!is_connected) {
        return false;
    }
    config.st_separate("path records");
    config.config_int("capture budget (MB)", budget_mb,
                      "Size cap for the path record stream consumed by a debugger.", 16, 4096);
    return stream_size(spp, vertices_per_path) != emitted_size;
}

vk::DeviceSize PathRecordSink::stream_size(const uint32_t spp,
                                           const uint32_t vertices_per_path) const {
    const uint64_t paths = static_cast<uint64_t>(last_extent.width) * last_extent.height * spp;
    const uint64_t budget = static_cast<uint64_t>(std::max(budget_mb, 1)) << 20;
    return std::min({path_record_buffer_size(paths, vertices_per_path), budget, last_max_range});
}

} // namespace merian
