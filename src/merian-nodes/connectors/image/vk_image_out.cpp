#include "merian-nodes/connectors/image/vk_image_out.hpp"

#include "merian-nodes/graph/errors.hpp"

namespace merian_nodes {

VkImageOut::VkImageOut(const std::string& name, const bool persistent, const uint32_t array_size)
    : OutputConnector(name, !persistent), persistent(persistent), array_size(array_size) {}

uint32_t VkImageOut::get_array_size() const {
    return array_size;
}

vk::ImageCreateInfo VkImageOut::get_create_info() const {
    throw graph_errors::invalid_connection{fmt::format(
        "This VkImageOut connector {} does not supply create infos for its images.", name)};
}

} // namespace merian_nodes
