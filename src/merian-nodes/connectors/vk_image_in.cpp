#include "merian-nodes/connectors/vk_image_in.hpp"

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
    : TypedInputConnector(name, delay, optional), access_flags(access_flags),
      pipeline_stages(pipeline_stages), required_layout(required_layout), usage_flags(usage_flags),
      stage_flags(stage_flags) {}

std::optional<vk::DescriptorSetLayoutBinding> VkImageIn::get_descriptor_info() const {
    if (stage_flags) {
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageImage, array_size,
                                              stage_flags, nullptr};
    }
    return std::nullopt;
}

void VkImageIn::get_descriptor_update(const uint32_t binding,
                                             const GraphResourceHandle& resource,
                                             const DescriptorSetHandle& update,
                                             const ResourceAllocatorHandle& allocator) {
    if (!resource) {
        // the optional connector was not connected
        update->queue_descriptor_write_image(binding, allocator->get_dummy_texture()->get_view(), 0,
                                               vk::ImageLayout::eShaderReadOnlyOptimal);
    } else {
        // or vk::ImageLayout::eShaderReadOnlyOptimal instead of required?
        image_views.clear();
        const auto& res = debugable_ptr_cast<ImageArrayResource>(resource);
        for (auto& update_idx : res->pending_updates) {
            ImageViewHandle view = ImageView::create(res->get_image(update_idx)->make_view_create_info(), res->get_image(update_idx));
            image_views.push_back(view);
            update->queue_descriptor_write_image(
                binding, view, update_idx,
                required_layout);
        }
    }
}

Connector::ConnectorStatusFlags VkImageIn::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    auto res = debugable_ptr_cast<ImageArrayResource>(resource);

    if (res->last_used_as_output) {
        for (const auto& image : res->images) {
            if (!image) {
                continue;
            }

            vk::ImageMemoryBarrier2 img_bar = image->barrier2(
                required_layout, res->current_access_flags, res->input_access_flags,
                res->current_stage_flags, res->input_stage_flags);
            image_barriers.push_back(img_bar);
        }
        res->current_stage_flags = res->input_stage_flags;
        res->current_access_flags = res->input_access_flags;
        res->last_used_as_output = false;
    } else {
        for (const auto& image : res->images) {
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
    auto casted_output = std::dynamic_pointer_cast<VkImageOut>(output);
    if (!casted_output) {
        throw graph_errors::invalid_connection{
            fmt::format("VkImageIn {} cannot recive from {}.", name, output->name)};
    }
    array_size = casted_output->array_size();
}

ImageArrayResource& VkImageIn::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<ImageArrayResource>(resource);
}

} // namespace merian_nodes
