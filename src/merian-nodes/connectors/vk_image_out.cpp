#include "merian-nodes/connectors/vk_image_out.hpp"


#include "merian-nodes/graph/node.hpp"

namespace merian_nodes {

VkImageOut::VkImageOut(const std::string& name, const bool persistent, const uint32_t image_count)
    : TypedOutputConnector(name, !persistent), images(image_count) {}

Connector::ConnectorStatusFlags VkImageOut::on_post_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    debugable_ptr_cast<ImageArrayResource>(resource)->last_used_as_output = true;
    return {};
}

ImageArrayResource& VkImageOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<ImageArrayResource>(resource);
}


} // namespace merian_nodes
