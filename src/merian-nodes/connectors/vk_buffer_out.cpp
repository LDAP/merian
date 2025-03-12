//
// Created by oschdi on 3/11/25.
//

#include "merian-nodes/connectors/vk_buffer_out.hpp"

namespace merian_nodes {

VkBufferOut::VkBufferOut(const std::string& name,
                                       const uint32_t array_size,
                                       const bool persistent)
    : TypedOutputConnector(name, !persistent), array_size(array_size) {}

} // namespace merian-nodes