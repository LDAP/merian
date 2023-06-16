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

template <BlitNodeMode mode> class BlitNode : public Node {
  public:
    BlitNode() {}

    void set_target(vk::Image dst_image,
                    vk::ImageLayout dst_in_layout,
                    vk::ImageLayout dst_out_layout,
                    vk::Extent3D dst_extent) {
        this->dst_image = dst_image;
        this->dst_in_layout = dst_in_layout;
        this->dst_out_layout = dst_out_layout;
        this->dst_extent = dst_extent;
    }

    virtual std::string name() {
        return "BlitNode";
    }

    virtual std::tuple<std::vector<NodeInputDescriptorImage>,
                       std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() {
        return {
            {
                NodeInputDescriptorImage("blit_src", vk::AccessFlagBits2::eTransferRead,
                                         vk::PipelineStageFlagBits2::eTransfer,
                                         vk::ImageLayout::eTransferSrcOptimal,
                                         vk::ImageUsageFlagBits::eTransferSrc),
            },
            {},
        };
    }

    virtual std::tuple<std::vector<NodeOutputDescriptorImage>,
                       std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<NodeOutputDescriptorImage>&,
                     const std::vector<NodeOutputDescriptorBuffer>&) {
        return {};
    }

    virtual void cmd_build(const vk::CommandBuffer&,
                           const std::vector<std::vector<ImageHandle>>&,
                           const std::vector<std::vector<BufferHandle>>&,
                           const std::vector<std::vector<ImageHandle>>&,
                           const std::vector<std::vector<BufferHandle>>&) {}

    virtual void cmd_process(const vk::CommandBuffer& cmd,
                             const uint64_t,
                             const uint32_t,
                             const std::vector<ImageHandle>& image_inputs,
                             const std::vector<BufferHandle>&,
                             const std::vector<ImageHandle>&,
                             const std::vector<BufferHandle>&) {
        assert(image_inputs.size() == 1);
        if (!dst_image) {
            return;
        }

        auto& src_image = image_inputs[0];

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
