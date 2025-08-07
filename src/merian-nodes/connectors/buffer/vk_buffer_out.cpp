#include "merian-nodes/connectors/buffer/vk_buffer_out.hpp"

#include "merian-nodes/graph/errors.hpp"

namespace merian_nodes {

VkBufferOut::VkBufferOut(const std::string& name, const bool persistent, const uint32_t array_size)
    : OutputConnector(name, !persistent), persistent(persistent), array_size(array_size) {}

uint32_t VkBufferOut::get_array_size() const {
    return array_size;
}

vk::BufferCreateInfo VkBufferOut::get_create_info() const {
    throw graph_errors::invalid_connection{fmt::format(
        "This VkBufferOut connector {} does not supply create infos for its buffers.", name)};
}

} // namespace merian_nodes
