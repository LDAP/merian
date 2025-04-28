#include "merian-nodes/connectors/vk_image_in.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"

namespace merian_nodes {

constexpr uint32_t MIN_DESCRIPTOR_COUNT = 1;

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
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eCombinedImageSampler, std::max(MIN_DESCRIPTOR_COUNT, array_size),
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
        update->queue_descriptor_write_texture(binding, allocator->get_dummy_texture(), 0,
                                               vk::ImageLayout::eShaderReadOnlyOptimal);
    } else {
        const auto& res = debugable_ptr_cast<ImageArrayResource>(resource);
        for (auto& pending_update : res->pending_updates) {
            const TextureHandle tex =
                res->textures[pending_update].has_value() ? res->textures[pending_update].value() : allocator->get_dummy_texture();
            update->queue_descriptor_write_texture(binding, tex, pending_update, required_layout);
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

    for (auto& image : res->images) {
        if (!image) {
            continue;
        }

        if (res->last_used_as_output) {
			vk::ImageMemoryBarrier2 img_bar =image->barrier2(
            	required_layout, res->current_access_flags, res->input_access_flags,
            	res->current_stage_flags, res->input_stage_flags);
        	image_barriers.push_back(img_bar);
     	    res->current_stage_flags = res->input_stage_flags;
     	    res->current_access_flags = res->input_access_flags;
       		res->last_used_as_output = false;
        } else {
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
        std::swap(res->pending_updates, res->current_updates);
        return NEEDS_DESCRIPTOR_UPDATE;
    }

    return flags;
}

void VkImageIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<VkImageOut>(output);
    if (!casted_output) {
        throw graph_errors::invalid_connection{
            fmt::format("VkImageIn {} cannot recive from {}.", name, output->name)};
    }
}

ImageArrayResource& VkImageIn::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<ImageArrayResource>(resource);
}

std::shared_ptr<VkImageIn>
VkImageIn::compute_read(const std::string& name, const uint32_t delay, const bool optional) {
    return std::make_shared<VkImageIn>(
        name, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageUsageFlagBits::eSampled,
        vk::ShaderStageFlagBits::eCompute, delay, optional);
}

std::shared_ptr<VkImageIn>
VkImageIn::transfer_src(const std::string& name, const uint32_t delay, const bool optional) {
    return std::make_shared<VkImageIn>(
        name, vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eAllTransfer,
        vk::ImageLayout::eTransferSrcOptimal, vk::ImageUsageFlagBits::eTransferSrc,
        vk::ShaderStageFlags(), delay, optional);
}

} // namespace merian_nodes
