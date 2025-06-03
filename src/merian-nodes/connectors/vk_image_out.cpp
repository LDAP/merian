#include "merian-nodes/connectors/managed_vk_image_out.hpp"

#include "merian-nodes/connectors/vk_texture_in.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"

namespace merian_nodes {

VkImageOut::VkImageOut(const std::string& name,
                      const bool persistent,
                      const uint32_t array_size) : TypedOutputConnector(name, !persistent),
                        persistent(persistent), images(array_size) {}

ImageArrayResource& VkImageOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<ImageArrayResource>(resource);
}

uint32_t VkImageOut::array_size() const {
    return images.size();
}
} // namespace merian_nodes
