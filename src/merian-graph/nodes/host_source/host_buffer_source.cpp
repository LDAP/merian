#include "merian-graph/nodes/host_source/host_buffer_source.hpp"

#include <cstdint>
#include <cstring>

namespace merian {

void HostBufferSource::initialize(const ContextHandle& context,
                                  const ResourceAllocatorHandle& allocator) {
    HostSource::initialize(context, allocator);
    upload(sizeof(uint32_t), [](void* data) { std::memset(data, 0, sizeof(uint32_t)); });
}

std::vector<OutputConnectorDescriptor>
HostBufferSource::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    create_output(
        [&] { con_out = ManagedVkBufferOut::create(vk::BufferCreateInfo{{}, size, {}}); });
    return {{.name = "out",
             .connector = con_out,
             .access = ConnectorAccess::transfer_dst,
             .persistent = true}};
}

void HostBufferSource::upload(const vk::DeviceSize size,
                              const std::function<void(void* data)>& fill) {
    stage_upload(size, fill, [&] {
        const bool changed = size != this->size;
        this->size = size;
        return changed;
    });
}

void HostBufferSource::cmd_copy(const CommandBufferHandle& cmd,
                                const NodeIO& io,
                                const BufferHandle& staging_buffer,
                                const vk::DeviceSize staging_offset) {
    cmd->copy(staging_buffer, io[con_out].get_buffer(0), vk::BufferCopy{staging_offset, 0, size});
}

} // namespace merian
