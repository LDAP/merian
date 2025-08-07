#include "merian-nodes/connectors/image/vk_image_in.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"

namespace merian_nodes {

VkImageIn::VkImageIn(const std::string& name,
                     const vk::AccessFlags2 access_flags,
                     const vk::PipelineStageFlags2 pipeline_stages,
                     const vk::ImageLayout required_layout,
                     const vk::ImageUsageFlags usage_flags,
                     const vk::ShaderStageFlags stage_flags,
                     const uint32_t delay,
                     const bool optional)
    : InputConnector(name, delay, optional), access_flags(access_flags),
      pipeline_stages(pipeline_stages), required_layout(required_layout), usage_flags(usage_flags),
      stage_flags(stage_flags) {}

Connector::ConnectorStatusFlags
VkImageIn::on_pre_process([[maybe_unused]] GraphRun& run,
                          [[maybe_unused]] const CommandBufferHandle& cmd,
                          const GraphResourceHandle& resource,
                          [[maybe_unused]] const NodeHandle& node,
                          std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                          [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    auto res = debugable_ptr_cast<ImageArrayResource>(resource);

    if (res->last_used_as_output) {
        for (uint32_t i = 0; i < get_array_size(); i++) {
            const auto& image = res->get_image(i);
            if (!image) {
                continue;
            }

            vk::ImageMemoryBarrier2 img_bar =
                image->barrier2(required_layout, res->current_access_flags, res->input_access_flags,
                                res->current_stage_flags, res->input_stage_flags);
            image_barriers.push_back(img_bar);
        }
        res->current_stage_flags = res->input_stage_flags;
        res->current_access_flags = res->input_access_flags;
        res->last_used_as_output = false;
    } else {
        for (uint32_t i = 0; i < get_array_size(); i++) {
            const auto& image = res->get_image(i);
            if (!image) {
                continue;
            }

            // No barrier required, if no transition required
            if (required_layout != image->get_current_layout()) {
                vk::ImageMemoryBarrier2 img_bar = image->barrier2(
                    required_layout, res->current_access_flags, res->current_access_flags,
                    res->current_stage_flags, res->current_stage_flags);
                image_barriers.push_back(img_bar);
            }
        }
    }

    Connector::ConnectorStatusFlags flags{};
    if (!res->current_updates.empty()) {
        res->pending_updates.clear();
        std::swap(res->current_updates, res->pending_updates);
        flags |= NEEDS_DESCRIPTOR_UPDATE;
    }

    return flags;
}

void VkImageIn::on_connect_output(const OutputConnectorHandle& output) {
    const auto casted_output = std::dynamic_pointer_cast<VkImageOut>(output);

    if (!casted_output) {
        throw graph_errors::invalid_connection{
            fmt::format("This connector ({}) cannot receive recive from {}. Only connectors "
                        "derived from VkImageOut are supported.",
                        name, output->name)};
    }

    array_size = casted_output->get_array_size();
}

const ImageArrayResource& VkImageIn::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<const ImageArrayResource>(resource);
}

std::shared_ptr<VkImageIn>
VkImageIn::transfer_src(const std::string& name, const uint32_t delay, const bool optional) {
    return std::make_shared<VkImageIn>(
        name, vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eAllTransfer,
        vk::ImageLayout::eTransferSrcOptimal, vk::ImageUsageFlagBits::eTransferSrc,
        vk::ShaderStageFlags(), delay, optional);
}

} // namespace merian_nodes
