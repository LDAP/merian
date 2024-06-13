#pragma once

#include "merian-nodes/graph/connector_output.hpp"

namespace merian_nodes {

class VkImageArrayOut;
using VkImageArrayOutHandle = std::shared_ptr<VkImageArrayOut>;

// Output an array of textures. 
class VkImageArrayOut : public TypedOutputConnector<ImageHandle> {

};

} // namespace merian_nodes
