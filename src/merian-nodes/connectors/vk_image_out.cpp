#include "vk_image_out.hpp"

#include "connectors/vk_image_in.hpp"
#include "graph/errors.hpp"

namespace merian_nodes {

VkImageOut::VkImageOut(const std::string& name,
                       const vk::AccessFlags2 access_flags,
                       const vk::PipelineStageFlags2 pipeline_stages,
                       const vk::ImageLayout required_layout,
                       const vk::ShaderStageFlags stage_flags,
                       const vk::ImageCreateInfo create_info)
    : TypedOutputConnector(name, true), access_flags(access_flags), pipeline_stages(pipeline_stages),
      required_layout(required_layout), stage_flags(stage_flags), create_info(create_info) {}

std::optional<vk::DescriptorSetLayoutBinding> VkImageOut::get_descriptor_info() const {
    return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageImage, 1, stage_flags, nullptr};
}

void VkImageOut::get_descriptor_update(const uint32_t binding,
                                       GraphResourceHandle& resource,
                                       DescriptorSetUpdate& update) {
    // or vk::ImageLayout::eGeneral instead of required?
    update.write_descriptor_texture(binding, this->resource(resource), 0, 1, required_layout);
}

Connector::ConnectorStatusFlags
VkImageOut::on_pre_process([[maybe_unused]] GraphRun& run,
                           [[maybe_unused]] const vk::CommandBuffer& cmd,
                           GraphResourceHandle& resource,
                           [[maybe_unused]] const NodeHandle& node,
                           std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                           [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    Connector::ConnectorStatusFlags flags{};
    const auto& res = debugable_ptr_cast<VkTextureResource>(resource);
    if (res->needs_descriptor_update) {
        flags |= NEEDS_DESCRIPTOR_UPDATE;
        res->needs_descriptor_update = false;
    }

    vk::ImageMemoryBarrier2 img_bar = res->tex->get_image()->barrier2(
        required_layout, res->current_access_flags, access_flags, res->current_stage_flags, pipeline_stages,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), true);

    image_barriers.push_back(img_bar);
    res->current_stage_flags = pipeline_stages;
    res->current_access_flags = access_flags;

    return flags;
}

Connector::ConnectorStatusFlags
VkImageOut::on_post_process([[maybe_unused]] GraphRun& run,
                            [[maybe_unused]] const vk::CommandBuffer& cmd,
                            GraphResourceHandle& resource,
                            [[maybe_unused]] const NodeHandle& node,
                            [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                            [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    debugable_ptr_cast<VkTextureResource>(resource)->last_used_as_output = true;
    return {};
}

GraphResourceHandle VkImageOut::create_resource(const SharedContext& context,
                                                const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                                                [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                                                const ResourceAllocatorHandle& aliasing_allocator) {
    vk::ImageCreateInfo create_info = this->create_info;
    vk::PipelineStageFlags2 input_pipeline_stages;
    vk::AccessFlags2 input_access_flags;

    for (auto& [input_node, input] : inputs) {
        const auto& image_in = std::dynamic_pointer_cast<VkImageIn>(input);
        if (!image_in) {
            throw graph_errors::connector_error{fmt::format("VkImageOut {} cannot output to {}.", name, input->name)};
        }
        create_info.usage |= image_in->usage_flags;
        input_pipeline_stages |= image_in->pipeline_stages;
        input_access_flags |= image_in->access_flags;
    }

    const ImageHandle image = aliasing_allocator->createImage(create_info, NONE);
    vk::ImageViewCreateInfo create_image_view{
        {}, *image, vk::ImageViewType::e2D, image->get_format(), {}, first_level_and_layer()};
    const TextureHandle tex = allocator->createTexture(image, create_image_view);

    // todo: make sampler configurable per output and input connector
    const vk::FormatProperties props =
        context->physical_device.physical_device.getFormatProperties(create_image_view.format);
    if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear) {
        tex->attach_sampler(allocator->get_sampler_pool()->linear_mirrored_repeat());
    } else {
        tex->attach_sampler(allocator->get_sampler_pool()->nearest_mirrored_repeat());
    }

    return std::make_shared<VkTextureResource>(tex, input_pipeline_stages, input_access_flags);
}

TextureHandle VkImageOut::resource(GraphResourceHandle& resource) {
    return debugable_ptr_cast<VkTextureResource>(resource)->tex;
}

VkImageOut VkImageOut::compute_write(const std::string& name, const vk::Format format, const vk::Extent3D extent) {
    const vk::ImageCreateInfo create_info{
        {},
        vk::ImageType::e2D,
        format,
        extent,
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };

    return VkImageOut{name,
                      vk::AccessFlagBits2::eShaderWrite,
                      vk::PipelineStageFlagBits2::eComputeShader,
                      vk::ImageLayout::eGeneral,
                      vk::ShaderStageFlagBits::eCompute,
                      create_info};
}

VkImageOut VkImageOut::compute_write(const std::string& name,
                                     const vk::Format format,
                                     const uint32_t width,
                                     const uint32_t height,
                                     const uint32_t depth) {
    return compute_write(name, format, {width, height, depth});
}

VkImageOut VkImageOut::compute_read_write(const std::string& name, const vk::Format format, const vk::Extent3D extent) {
    const vk::ImageCreateInfo create_info{
        {},
        vk::ImageType::e2D,
        format,
        extent,
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };

    return VkImageOut{
        name,
        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageLayout::eGeneral,
        vk::ShaderStageFlagBits::eCompute,
        create_info,
    };
}

VkImageOut VkImageOut::transfer_write(const std::string& name, const vk::Format format, const vk::Extent3D extent) {
    const vk::ImageCreateInfo create_info{
        {},
        vk::ImageType::e2D,
        format,
        extent,
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };

    return VkImageOut{
        name,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageLayout::eTransferDstOptimal,
        {},
        create_info,
    };
}

VkImageOut VkImageOut::transfer_write(const std::string& name,
                                      const vk::Format format,
                                      const uint32_t width,
                                      const uint32_t height,
                                      const uint32_t depth) {
    return transfer_write(name, format, {width, height, depth});
}

} // namespace merian_nodes
