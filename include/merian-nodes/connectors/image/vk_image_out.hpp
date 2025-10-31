#pragma once

#include "merian-nodes/graph/connector_output.hpp"

namespace merian {

class VkImageOut;
using VkImageOutHandle = std::shared_ptr<VkImageOut>;

class VkImageOut : public OutputConnector {
  public:
    VkImageOut(const std::string& name,
               const bool persistent = false,
               const uint32_t array_size = 1);

    uint32_t get_array_size() const;

    virtual std::optional<vk::ImageCreateInfo> get_create_info(const uint32_t index = 0) const;

    // Throws node_error if create infos were not supplied.
    vk::ImageCreateInfo get_create_info_or_throw(const uint32_t index = 0) const;

  public:
    const bool persistent;

  private:
    const uint32_t array_size;
};

} // namespace merian
