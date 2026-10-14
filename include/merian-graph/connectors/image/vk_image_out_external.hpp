#pragma once

#include "vk_image_out.hpp"

#include "merian-graph/resources/image_array_resource_external.hpp"

namespace merian {

class ExternalVkImageOut;
using ExternalVkImageOutHandle = std::shared_ptr<ExternalVkImageOut>;

// Outputs an image that the node sets every iteration (e.g. a swapchain image), or none.
// Consumers must handle get_image() returning null.
class ExternalVkImageOut : public VkImageOut,
                           public AccessibleConnector<ExternalImageArrayResource&> {
  public:
    ExternalVkImageOut();

    GraphResourceHandle create_resource(const ResourceCreateInfo& info) override;

    bool supports_delay() const override {
        return false;
    }

    ExternalImageArrayResource& resource(const GraphResourceHandle& resource) override;

  public:
    static ExternalVkImageOutHandle create();
};

} // namespace merian
