//
// Created by oschdi on 3/11/25.
//

#ifndef VKBUFFEROUT_HPP
#define VKBUFFEROUT_HPP

#include "merian-nodes/graph/connector_output.hpp"

namespace merian_nodes {

class VkBufferOut : public TypedOutputConnector<BufferHandle> {
public:
    VkBufferOut(const std::string& name, const uint32_t array_size = 1, const bool persistent = false);

protected:
    uint32_t array_size;
};

} // namespace merian-nodes

#endif //VKBUFFEROUT_HPP
