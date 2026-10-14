#pragma once

#include "merian-graph/graph/connector_output.hpp"

namespace merian {

class VkImageOut;
using VkImageOutHandle = std::shared_ptr<VkImageOut>;

class VkImageOut : public OutputConnector {
  public:
    VkImageOut(const uint32_t array_size = 1);

    bool supports_in_place() const override {
        return true;
    }

    uint32_t get_array_size() const;

    virtual std::optional<vk::ImageCreateInfo> get_create_info(const uint32_t index = 0) const;

    // Throws node_error if create infos were not supplied.
    vk::ImageCreateInfo get_create_info_or_throw(const uint32_t index = 0) const;

  private:
    const uint32_t array_size;
};

} // namespace merian
