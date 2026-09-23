#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-shaders/debug/path_record.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/utils/properties.hpp"

#include <limits>

namespace merian {

// The renderer side of the path record stream; its header assumes a single consumer.
class PathRecordSink {
  public:
    OutputConnectorDescriptor describe_output(const ContextHandle& context,
                                              const vk::Extent3D& extent,
                                              uint32_t spp,
                                              uint32_t vertices_per_path);

    // true when connectivity changed; the stream is sized at the reconnect this requires
    [[nodiscard]] bool update_connected(const NodeIOLayout& io_layout);

    void add_constants(const SlangCompositionHandle& composition) const;

    // call once per process()
    void process(const CommandBufferHandle& cmd,
                 const NodeIO& io,
                 uint64_t iteration,
                 ShaderCursor cursor) const;

    // binds the stream for further dispatches of the same process()
    void bind(const NodeIO& io, ShaderCursor cursor) const;

    bool connected() const {
        return is_connected;
    }

    [[nodiscard]] bool properties(Properties& config, uint32_t spp, uint32_t vertices_per_path);

  private:
    vk::DeviceSize stream_size(uint32_t spp, uint32_t vertices_per_path) const;

    ManagedVkBufferOutHandle con;
    bool is_connected = false;
    int32_t budget_mb = 256;
    vk::DeviceSize emitted_size = 0;
    vk::DeviceSize last_max_range = std::numeric_limits<vk::DeviceSize>::max();
    vk::Extent3D last_extent{};
};

} // namespace merian
