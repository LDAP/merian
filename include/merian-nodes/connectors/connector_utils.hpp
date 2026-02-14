#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"

namespace merian {

using GBufferOutHandle = ManagedVkImageOutHandle;
using GBufferInHandle = VkSampledImageInHandle;

class GBufferOut {
  public:
    static GBufferOutHandle compute_write(const uint32_t width, const uint32_t height) {
        return ManagedVkImageOut::compute_write(vk::Format::eR32G32B32A32Uint, width, height);
    }
};

class GBufferIn {
  public:
    static GBufferInHandle compute_read(const uint32_t delay = 0, const bool optional = false) {
        return VkSampledImageIn::compute_read(delay, optional);
    }
};

} // namespace merian
