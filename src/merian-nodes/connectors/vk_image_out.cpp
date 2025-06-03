#include "merian-nodes/connectors/managed_vk_image_out.hpp"

#include "merian-nodes/connectors/vk_texture_in.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"

namespace merian_nodes {

VkImageOut::VkImageOut(const std::string& name,
                      const bool persistent,
                      const uint32_t array_size) : TypedOutputConnector(name, !persistent),
                        persistent(persistent), images(array_size) {}

VkImageOut::VkImageOut(const std::string& name,
                      const vk::ImageCreateInfo create_info,
                      const bool persistent,
                      const uint32_t array_size) : TypedOutputConnector(name, !persistent),
                        create_info(create_info), persistent(persistent), images(array_size) {}

ImageArrayResource& VkImageOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<ImageArrayResource>(resource);
}

uint32_t VkImageOut::array_size() const {
    return images.size();
}

vk::ImageCreateInfo VkImageOut::get_create_info() const {
    if (create_info.has_value()) {
        return create_info.value();
    }

    throw graph_errors::invalid_connection{
            fmt::format("VkImageIn {} has no set create info", name)};
}
} // namespace merian_nodes
