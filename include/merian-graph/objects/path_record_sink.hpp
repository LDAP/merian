#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-shaders/debug/path_record.hpp"
#include "merian/utils/properties.hpp"

#include <limits>

namespace merian {

// The renderer side of the path record stream: a member plus one call per lifecycle method makes
// a renderer debuggable, and the shader hooks behind merian_path_record compile away while no
// debugger consumes the output.
class PathRecordSink {
  public:
    OutputConnectorDescriptor
    describe_output(const vk::Extent3D& extent,
                    const uint32_t spp,
                    const uint32_t max_path_length,
                    const vk::DeviceSize max_range = std::numeric_limits<vk::DeviceSize>::max()) {
        vk::DeviceSize size = vk::DeviceSize{PATH_RECORD_HEADER_UINTS} * 4;
        if (is_connected) {
            const uint64_t paths = static_cast<uint64_t>(extent.width) * extent.height * spp;
            size = std::min(path_record_buffer_size(paths, max_path_length + 1),
                            static_cast<uint64_t>(budget_mb) << 20);
            // descriptors beyond maxStorageBufferRange are undefined on drivers with raw
            // pointer + size descriptors
            size = std::min(size, max_range & ~vk::DeviceSize{3});
        }
        last_max_range = max_range;
        emitted_size = size;
        last_extent = extent;
        last_spp = spp;
        last_max_path_length = max_path_length;
        con = ManagedVkBufferOut::create(vk::BufferCreateInfo(
            {}, size,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc));
        return {"path_records", con, ConnectorAccess::ray_tracing_read_write};
    }

    // The graph hands out uninitialized memory; a garbage header must not arm recording.
    void begin_frame(const CommandBufferHandle& cmd, const NodeIO& io, const uint64_t iteration) {
        if (is_connected && iteration == 0) {
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
    }

    // Connectivity is only visible once connected, so a change has to round-trip through
    // NEEDS_RECONNECT before describe_output can size the stream.
    [[nodiscard]] bool update_connected(const NodeIOLayout& io_layout) {
        if (io_layout.is_connected(con) == is_connected) {
            return false;
        }
        is_connected = !is_connected;
        return true;
    }

    bool connected() const {
        return is_connected;
    }

    // Appended to the renderer's link-time constants module (inside or outside namespace merian).
    std::string export_line() const {
        return is_connected ? "namespace merian { export static const bool merian_path_record = "
                              "true; }"
                            : "namespace merian { export static const bool merian_path_record = "
                              "false; }";
    }

    void bind(ShaderCursor cursor, const NodeIO& io) const {
        if (cursor.is_valid()) {
            cursor = static_cast<const BufferHandle&>(io[con]);
        }
    }

    // Renders the capture budget; true if the stream has to be resized.
    [[nodiscard]] bool properties(Properties& config) {
        if (!is_connected) {
            return false;
        }
        config.st_separate("path records");
        config.config_int("capture budget (MB)", budget_mb,
                          "Size cap for the path record stream consumed by a debugger.", 16, 4096);
        const uint64_t paths =
            static_cast<uint64_t>(last_extent.width) * last_extent.height * last_spp;
        const vk::DeviceSize size =
            std::min({path_record_buffer_size(paths, last_max_path_length + 1),
                      static_cast<uint64_t>(budget_mb) << 20, last_max_range & ~vk::DeviceSize{3}});
        return size != emitted_size;
    }

    // Sizing inputs that changed since describe_output (e.g. spp edits) also need a reconnect.
    [[nodiscard]] bool needs_resize(const uint32_t spp, const uint32_t max_path_length) const {
        if (!is_connected) {
            return false;
        }
        return spp != last_spp || max_path_length != last_max_path_length;
    }

  private:
    ManagedVkBufferOutHandle con;
    bool is_connected = false;
    int32_t budget_mb = 256;
    vk::DeviceSize emitted_size = 0;
    vk::DeviceSize last_max_range = std::numeric_limits<vk::DeviceSize>::max();
    vk::Extent3D last_extent{};
    uint32_t last_spp = 1;
    uint32_t last_max_path_length = 8;
};

} // namespace merian
