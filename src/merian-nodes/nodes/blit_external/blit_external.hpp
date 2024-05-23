#pragma once

#include "merian-nodes/graph/node.hpp"
#include "merian/vk/utils/barriers.hpp"
#include "merian/vk/utils/blits.hpp"

namespace merian_nodes {

/**
 * Blits an image from the graph to an external user-supplied image.
 */
template <BlitNodeMode mode> class BlitExternalNode : public Node {
  public:
    BlitExternalNode() : Node("BlitExternalNode") {}

    ~BlitExternalNode() {}

    void set_target(vk::Image dst_image,
                    vk::ImageLayout dst_in_layout,
                    vk::ImageLayout dst_out_layout,
                    vk::Extent3D dst_extent) {
        this->dst_image = dst_image;
        this->dst_in_layout = dst_in_layout;
        this->dst_out_layout = dst_out_layout;
        this->dst_extent = dst_extent;
    }

    

    virtual void cmd_process(const vk::CommandBuffer& cmd,
                             [[maybe_unused]] GraphRun& run,
                             [[maybe_unused]] const std::shared_ptr<FrameData>& frame_data,
                             [[maybe_unused]] const uint32_t set_index,
                             const NodeIO& io) override {
        assert(io.image_inputs.size() == 1);
        if (!dst_image) {
            return;
        }
    }

  private:
    vk::Image dst_image;
    vk::ImageLayout dst_in_layout;
    vk::ImageLayout dst_out_layout;
    vk::Extent3D dst_extent;
};

} // namespace merian
