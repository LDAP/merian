#pragma once

#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/connectors/managed_vk_image_out.hpp"

namespace merian_nodes {

using GBufferOutHandle = ManagedVkImageOutHandle;
using GBufferInHandle = ManagedVkImageInHandle;

class GBufferOut {
  public:
    static GBufferOutHandle
    compute_write(const std::string& name, const uint32_t width, const uint32_t height) {
        return merian_nodes::ManagedVkImageOut::compute_write(name, vk::Format::eR32G32B32A32Uint,
                                                              width, height);
    }
};

class GBufferIn {
  public:
    static GBufferInHandle
    compute_read(const std::string& name, const uint32_t delay = 0, const bool optional = false) {
        return merian_nodes::ManagedVkImageIn::compute_read(name, delay, optional);
    }
};

} // namespace merian_nodes
