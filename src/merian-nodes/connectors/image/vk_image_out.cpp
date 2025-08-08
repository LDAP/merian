#include "merian-nodes/connectors/image/vk_image_out.hpp"
#include "merian-nodes/graph/errors.hpp"

namespace merian_nodes {

VkImageOut::VkImageOut(const std::string& name, const bool persistent, const uint32_t array_size)
    : OutputConnector(name, !persistent), persistent(persistent), array_size(array_size) {}

uint32_t VkImageOut::get_array_size() const {
    return array_size;
}

std::optional<vk::ImageCreateInfo> VkImageOut::get_create_info(const uint32_t /*index*/) const {
    return std::nullopt;
}

// Throws node_error if create infos were not supplied.
vk::ImageCreateInfo VkImageOut::get_create_info_or_throw(const uint32_t index) const {
    std::optional<vk::ImageCreateInfo> optional_infos = get_create_info(index);
    if (!optional_infos.has_value()) {
        throw graph_errors::node_error{
            fmt::format("create infos were not provided by connector {}", name)};
    }
    return *optional_infos;
}

} // namespace merian_nodes
