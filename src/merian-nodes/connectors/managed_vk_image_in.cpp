#include "merian-nodes/connectors/managed_vk_image_in.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-nodes/resources/managed_vk_image_resource.hpp"

namespace merian_nodes {

ManagedVkImageIn::ManagedVkImageIn(const std::string& name,
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

std::optional<vk::DescriptorSetLayoutBinding> ManagedVkImageIn::get_descriptor_info() const {
    if (stage_flags) {
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eCombinedImageSampler, 1,
                                              stage_flags, nullptr};
    }
    return std::nullopt;
}

void ManagedVkImageIn::get_descriptor_update(const uint32_t binding,
                                             const GraphResourceHandle& resource,
                                             const DescriptorSetHandle& update,
                                             const ResourceAllocatorHandle& allocator) {
    if (!resource) {
        // the optional connector was not connected
        update->queue_descriptor_write_texture(binding, allocator->get_dummy_texture(), 0,
                                               vk::ImageLayout::eShaderReadOnlyOptimal);
    } else {
        // or vk::ImageLayout::eShaderReadOnlyOptimal instead of required?
        assert(debugable_ptr_cast<ManagedVkImageResource>(resource)->tex && "missing usage flags?");
        update->queue_descriptor_write_texture(
            binding, *debugable_ptr_cast<ManagedVkImageResource>(resource)->tex, 0,
            required_layout);
    }
}

Connector::ConnectorStatusFlags ManagedVkImageIn::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    auto res = debugable_ptr_cast<ManagedVkImageResource>(resource);

    if (res->last_used_as_output) {
        vk::ImageMemoryBarrier2 img_bar = res->image->barrier2(
            required_layout, res->current_access_flags, res->input_access_flags,
            res->current_stage_flags, res->input_stage_flags);
        image_barriers.push_back(img_bar);
        res->current_stage_flags = res->input_stage_flags;
        res->current_access_flags = res->input_access_flags;
        res->last_used_as_output = false;
    } else {
        // No barrier required, if no transition required
        if (required_layout != res->image->get_current_layout()) {
            vk::ImageMemoryBarrier2 img_bar = res->image->barrier2(
                required_layout, res->current_access_flags, res->current_access_flags,
                res->current_stage_flags, res->current_stage_flags);
            image_barriers.push_back(img_bar);
        }
    }

    Connector::ConnectorStatusFlags flags{};
    if (res->needs_descriptor_update) {
        flags |= NEEDS_DESCRIPTOR_UPDATE;
        res->needs_descriptor_update = false;
    }

    return flags;
}

void ManagedVkImageIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<ManagedVkImageOut>(output);
    if (!casted_output) {
        throw graph_errors::invalid_connection{
            fmt::format("ManagedVkImageIn {} cannot recive from {}.", name, output->name)};
    }
}

ImageHandle ManagedVkImageIn::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<ManagedVkImageResource>(resource)->image;
}

std::shared_ptr<ManagedVkImageIn>
ManagedVkImageIn::compute_read(const std::string& name, const uint32_t delay, const bool optional) {
    return std::make_shared<ManagedVkImageIn>(
        name, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageUsageFlagBits::eSampled,
        vk::ShaderStageFlagBits::eCompute, delay, optional);
}

std::shared_ptr<ManagedVkImageIn> ManagedVkImageIn::fragment_read(const std::string& name,
                                                                  const uint32_t delay,
                                                                  const bool optional) {
    return std::make_shared<ManagedVkImageIn>(
        name, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageUsageFlagBits::eSampled,
        vk::ShaderStageFlagBits::eFragment, delay, optional);
}

std::shared_ptr<ManagedVkImageIn>
ManagedVkImageIn::transfer_src(const std::string& name, const uint32_t delay, const bool optional) {
    return std::make_shared<ManagedVkImageIn>(
        name, vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eAllTransfer,
        vk::ImageLayout::eTransferSrcOptimal, vk::ImageUsageFlagBits::eTransferSrc,
        vk::ShaderStageFlags(), delay, optional);
}

} // namespace merian_nodes
