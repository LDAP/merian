#include "merian-graph/connectors/vk_tlas_out.hpp"
#include "merian-graph/connectors/vk_tlas_in.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian/utils/pointer.hpp"

namespace merian {

VkTLASOut::VkTLASOut() : OutputConnector(true) {}

GraphResourceHandle VkTLASOut::create_resource(
    [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    [[maybe_unused]] const ConnectorAccess& combined_access,
    [[maybe_unused]] const ResourceAllocatorHandle& allocator,
    [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resource_index,
    [[maybe_unused]] const uint32_t ring_size) {
    return std::make_shared<TLASResource>();
}

TLASResource& VkTLASOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<TLASResource>(resource);
}

Connector::ConnectorStatusFlags
VkTLASOut::on_pre_process([[maybe_unused]] GraphRun& run,
                          [[maybe_unused]] const CommandBufferHandle& cmd,
                          const GraphResourceHandle& resource,
                          [[maybe_unused]] const NodeHandle& node,
                          [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                          [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<TLASResource>(resource);
    res->tlas.reset();

    return {};
}

Connector::ConnectorStatusFlags VkTLASOut::on_post_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<TLASResource>(resource);

    if (!res->tlas) {
        throw graph_errors::connector_error{"Node must set the TLAS for connector"};
    }

    return {};
}

VkTLASOutHandle VkTLASOut::create() {
    return std::make_shared<VkTLASOut>();
}

} // namespace merian
