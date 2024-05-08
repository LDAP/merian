#pragma once

#include "merian/vk/graph/node.hpp"
#include "merian/vk/utils/barriers.hpp"
#include "merian/vk/utils/blits.hpp"

namespace merian {

enum BlitNodeMode {
    FIT,
    FILL,
    STRETCH,
};

/**
 * Blits an image from the graph to an external user-supplied image.
 */
template <BlitNodeMode mode> class BlitExternalNode : public Node {
  public:
    BlitExternalNode() {}

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

    virtual std::string name() override {
        return "BlitExternalNode";
    }

    virtual std::tuple<std::vector<NodeInputDescriptorImage>,
                       std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override {
        return {
            {
                NodeInputDescriptorImage::transfer_src("src"),
            },
            {},
        };
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

        auto& src_image = io.image_inputs[0];

        if (dst_in_layout != vk::ImageLayout::eTransferDstOptimal)
            cmd_barrier_image_layout(cmd, dst_image, dst_in_layout,
                                     vk::ImageLayout::eTransferDstOptimal);

        if constexpr (mode == FIT) {
            cmd_blit_fit(cmd, *src_image, vk::ImageLayout::eTransferSrcOptimal,
                         src_image->get_extent(), dst_image, vk::ImageLayout::eTransferDstOptimal,
                         dst_extent);
        } else if constexpr (mode == FILL) {
            cmd_blit_fill(cmd, *src_image, vk::ImageLayout::eTransferSrcOptimal,
                          src_image->get_extent(), dst_image, vk::ImageLayout::eTransferDstOptimal,
                          dst_extent);
        } else if constexpr (mode == STRETCH) {
            cmd_blit_stretch(cmd, *src_image, vk::ImageLayout::eTransferSrcOptimal,
                             src_image->get_extent(), dst_image,
                             vk::ImageLayout::eTransferDstOptimal, dst_extent);
        }

        if (dst_out_layout != vk::ImageLayout::eTransferDstOptimal)
            cmd_barrier_image_layout(cmd, dst_image, vk::ImageLayout::eTransferDstOptimal,
                                     dst_out_layout);
    }

  private:
    vk::Image dst_image;
    vk::ImageLayout dst_in_layout;
    vk::ImageLayout dst_out_layout;
    vk::Extent3D dst_extent;
};

} // namespace merian
