#pragma once

#include "merian-graph/graph/graph_shader_object.hpp"
#include "merian-shaders/gbuffer.hpp"

namespace merian {

// GBuffer transported as a graph object: the graph allocates the four textures per ring slot
// and keeps them synchronized; consumers bind merian::GBuffer / merian::WGBuffer.
class GBufferObject : public GraphShaderObject {
    static constexpr uint32_t tex_count = 4;

  public:
    GBufferObject(const vk::Extent3D extent) : extent(extent) {}

    void allocate(const ShaderObjectAllocateInfo& info) override {
        constexpr std::array<vk::Format, tex_count> formats = {
            vk::Format::eR32G32B32A32Uint,   // surface
            vk::Format::eR32G32B32A32Uint,   // hit info
            vk::Format::eR16G16Sfloat,       // motion vectors
            vk::Format::eR32G32B32A32Sfloat, // albedo
        };

        gbuffer =
            std::make_shared<GBuffer>(info.compile_context, info.context, info.allocator, extent);

        std::array<ImageViewHandle, formats.size()> views;
        for (uint32_t i = 0; i < formats.size(); i++) {
            const vk::ImageCreateInfo create_info{
                {},
                vk::ImageType::e2D,
                formats[i],
                extent,
                1,
                1,
                vk::SampleCountFlagBits::e1,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                    vk::ImageUsageFlagBits::eTransferDst | info.combined_access.image_usage,
            };
            images[i] = info.allocator->create_image(create_info, MemoryMappingType::NONE);
            views[i] =
                info.allocator->create_image_view(images[i], images[i]->make_view_create_info());
        }
        gbuffer->set_resources(views[0], views[1], views[2], views[3]);
    }

    const ShaderObjectHandle& object(const ShaderAccess access) const override {
        return access == ShaderAccess::READ ? gbuffer->get_shader_object()
                                            : gbuffer->get_write_shader_object();
    }

    // The textures always live in eGeneral: the gbuffer bundles sampled reads and storage
    // writes. Cleared so consumers never see garbage (out-of-bounds ids in a gbuffer hang the
    // GPU).
    void on_connected(const CommandBufferHandle& cmd) override {
        constexpr vk::AccessFlags2 all_access =
            vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead;
        constexpr vk::PipelineStageFlags2 all_stages = vk::PipelineStageFlagBits2::eAllCommands;

        std::vector<vk::ImageMemoryBarrier2> to_transfer;
        to_transfer.reserve(images.size());
        for (const ImageHandle& image : images) {
            to_transfer.push_back(image->barrier2(vk::ImageLayout::eTransferDstOptimal, all_access,
                                                  vk::AccessFlagBits2::eTransferWrite, all_stages,
                                                  vk::PipelineStageFlagBits2::eAllTransfer));
        }
        cmd->barrier({}, {}, to_transfer);
        std::vector<vk::ImageMemoryBarrier2> to_general;
        to_general.reserve(images.size());
        for (const ImageHandle& image : images) {
            cmd->clear(image, vk::ImageLayout::eTransferDstOptimal);
            to_general.push_back(
                image->barrier2(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eTransferWrite,
                                all_access, vk::PipelineStageFlagBits2::eAllTransfer, all_stages));
        }
        cmd->barrier({}, {}, to_general);
    }

    vk::Extent3D get_extent() const {
        return extent;
    }

    const GBufferHandle& get_gbuffer() const {
        return gbuffer;
    }

  private:
    const vk::Extent3D extent;

    GBufferHandle gbuffer;
    std::array<ImageHandle, tex_count> images;
};

} // namespace merian
