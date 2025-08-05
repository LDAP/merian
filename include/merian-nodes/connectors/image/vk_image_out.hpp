#pragma once

#include "merian-nodes/graph/connector_output.hpp"

namespace merian_nodes {

class VkImageOut;
using VkImageOutHandle = std::shared_ptr<VkImageOut>;

class VkImageOut : public OutputConnector {
  public:
    VkImageOut(const std::string& name,
               const bool persistent = false,
               const uint32_t array_size = 1);

    uint32_t array_size() const;

    // Throws invalid_connection if create infos were not supplied.
    virtual vk::ImageCreateInfo get_create_info() const;

  public:
    const bool persistent;

  private:
    const uint32_t m_array_size;
};

} // namespace merian_nodes
