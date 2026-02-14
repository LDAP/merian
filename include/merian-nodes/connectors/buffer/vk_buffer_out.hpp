#pragma once

#include "merian-nodes/graph/connector_output.hpp"

namespace merian {

class VkBufferOut;
using VkBufferOutHandle = std::shared_ptr<VkBufferOut>;

class VkBufferOut : public OutputConnector {
  public:
    VkBufferOut(const bool persistent = false, const uint32_t array_size = 1);

    uint32_t get_array_size() const;

    virtual std::optional<vk::BufferCreateInfo> get_create_info(const uint32_t index = 0) const;

    // Throws node_error if create infos were not supplied.
    vk::BufferCreateInfo get_create_info_or_throw(const uint32_t index = 0) const;

  public:
    const bool persistent;

  private:
    const uint32_t array_size;
};

} // namespace merian
