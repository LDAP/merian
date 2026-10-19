#include "merian-graph/nodes/host_sink/host_buffer_sink.hpp"

#include <algorithm>

namespace merian {

std::vector<InputConnectorDescriptor> HostBufferSink::describe_inputs() {
    return {{.name = "src", .connector = con_src, .access = ConnectorAccess::transfer_src}};
}

HostBufferSink::NodeStatusFlags HostBufferSink::properties(Properties& config) {
    config.config_uint64("offset", offset, "Of the bytes captured.");
    config.config_uint64("size", size, "Bytes captured, 0 to the end.");
    config.config_uint("element", element, "Of a buffer array.");
    capture_properties(config);
    return {};
}

bool HostBufferSink::capturable(const NodeIO& io) const {
    return captured_range(io).second > 0;
}

HostCopy HostBufferSink::cmd_capture(const CommandBufferHandle& cmd,
                                     const NodeIO& io,
                                     [[maybe_unused]] const NodeProcessInfo& info) {
    const auto [buffer, captured_size] = captured_range(io);
    return cmd_copy_to_host(cmd, io.get_allocator()->get_staging(), buffer, offset, captured_size);
}

std::pair<BufferHandle, vk::DeviceSize> HostBufferSink::captured_range(const NodeIO& io) const {
    const BufferArrayResource& buffers = io[con_src];
    if (buffers.get_array_size() == 0) {
        return {nullptr, 0};
    }
    const BufferHandle& buffer =
        buffers.get_buffer(std::min(element, buffers.get_array_size() - 1));
    const vk::DeviceSize available =
        buffer && buffer->get_size() > offset ? buffer->get_size() - offset : 0;
    return {buffer, size == 0 ? available : std::min(size, available)};
}

} // namespace merian
