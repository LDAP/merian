#pragma once

#include "merian-nodes/graph/connector_output.hpp"

namespace merian_nodes {

class VkBufferOut;
using VkBufferOutHandle = std::shared_ptr<VkBufferOut>;

class VkBufferOut : public OutputConnector {
  public:
    VkBufferOut(const std::string& name,
                const bool persistent = false,
                const uint32_t array_size = 1);

    uint32_t get_array_size() const;

    // Throws invalid_connection if create infos were not supplied.
    virtual vk::BufferCreateInfo get_create_info() const;

  public:
    const bool persistent;

  private:
    const uint32_t array_size;
};

} // namespace merian_nodes
