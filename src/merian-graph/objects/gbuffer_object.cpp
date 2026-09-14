#include "merian-graph/objects/gbuffer_object.hpp"

#include "merian/utils/vector.hpp"

#include <fmt/format.h>

namespace merian {

GBufferObject::GBufferObject(const CreateInfo& create_info)
    : extent(create_info.extent),
      layout(create_info.layout ? create_info.layout : GBufferLayout::complete()) {}

void GBufferObject::allocate(const ShaderObjectAllocateInfo& info) {
    gbuffer = std::make_shared<GBuffer>(info.compile_context, info.context, info.allocator, extent,
                                        layout);

    for (uint32_t i = 0; i < layout->get_textures().size(); i++) {
        const vk::ImageCreateInfo create_info{
            {},
            vk::ImageType::e2D,
            layout->get_format(i, info.context->get_physical_device()),
            extent,
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                vk::ImageUsageFlagBits::eTransferDst | info.combined_access.image_usage,
        };
        const ImageHandle& image = images.emplace_back(info.allocator->create_image(
            create_info, MemoryMappingType::NONE, fmt::format("gbuffer tex{}", i)));
        views.emplace_back(
            info.allocator->create_image_view(image, image->make_view_create_info()));
    }
    gbuffer->set_resources(views);
}

const ShaderObjectHandle& GBufferObject::object(const ShaderAccess access) const {
    return access == ShaderAccess::READ ? gbuffer->get_shader_object()
                                        : gbuffer->get_write_shader_object();
}

void GBufferObject::on_connected(Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
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
        to_general.push_back(image->barrier2(vk::ImageLayout::eGeneral,
                                             vk::AccessFlagBits2::eTransferWrite, all_access,
                                             vk::PipelineStageFlagBits2::eAllTransfer, all_stages));
    }
    cmd->barrier({}, {}, to_general);
}

const ImageViewHandle& GBufferObject::get_view(const GBufferField field) const {
    return views[layout->get_texture_index(field)];
}

GBufferOut::GBufferOut(const vk::Extent3D& extent, const bool persistent)
    : ShaderObjectOut({.extent = extent}, persistent) {}

GBufferOutHandle GBufferOut::create(const vk::Extent3D& extent, const bool persistent) {
    return std::make_shared<GBufferOut>(extent, persistent);
}

std::shared_ptr<GBufferObject> GBufferOut::create_instance(
    const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs) {
    if (!layout) {
        std::vector<GBufferGroup> groups;
        bool complete = false;
        for (const auto& [node, input] : inputs) {
            if (const auto gbuffer_in = std::dynamic_pointer_cast<GBufferIn>(input)) {
                insert_all(groups, gbuffer_in->get_groups());
            } else {
                complete = true;
            }
        }
        layout =
            complete ? GBufferLayout::complete() : std::make_shared<const GBufferLayout>(groups);
    }
    return std::make_shared<GBufferObject>(
        GBufferObject::CreateInfo{.extent = get_create_info().extent, .layout = layout});
}

} // namespace merian
