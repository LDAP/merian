#include "vk_image_in.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-nodes/resources/vk_image_resource.hpp"

namespace merian_nodes {

VkImageIn::VkImageIn(const std::string& name,
                     const vk::AccessFlags2 access_flags,
                     const vk::PipelineStageFlags2 pipeline_stages,
                     const vk::ImageLayout required_layout,
                     const vk::ImageUsageFlags usage_flags,
                     const vk::ShaderStageFlags stage_flags,
                     const uint32_t delay)
    : TypedInputConnector(name, delay), access_flags(access_flags),
      pipeline_stages(pipeline_stages), required_layout(required_layout), usage_flags(usage_flags),
      stage_flags(stage_flags) {}

std::optional<vk::DescriptorSetLayoutBinding> VkImageIn::get_descriptor_info() const {
    if (stage_flags) {
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eCombinedImageSampler, 1,
                                              stage_flags, nullptr};
    } else {
        return std::nullopt;
    }
}

void VkImageIn::get_descriptor_update(const uint32_t binding,
                                      GraphResourceHandle& resource,
                                      DescriptorSetUpdate& update) {
    // // or vk::ImageLayout::eShaderReadOnlyOptimal instead of required?
    update.write_descriptor_texture(binding, this->resource(resource), 0, 1, required_layout);
}

Connector::ConnectorStatusFlags
VkImageIn::on_pre_process([[maybe_unused]] GraphRun& run,
                          [[maybe_unused]] const vk::CommandBuffer& cmd,
                          GraphResourceHandle& resource,
                          [[maybe_unused]] const NodeHandle& node,
                          std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                          [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    auto res = debugable_ptr_cast<VkImageResource>(resource);

    if (res->last_used_as_output) {
        vk::ImageMemoryBarrier2 img_bar = res->tex->get_image()->barrier2(
            required_layout, res->current_access_flags, res->input_access_flags,
            res->current_stage_flags, res->input_stage_flags);
        image_barriers.push_back(img_bar);
        res->current_stage_flags = res->input_stage_flags;
        res->current_access_flags = res->input_access_flags;
        res->last_used_as_output = false;
    } else {
        // No barrier required, if no transition required
        if (required_layout != res->tex->get_current_layout()) {
            vk::ImageMemoryBarrier2 img_bar = res->tex->get_image()->barrier2(
                required_layout, res->current_access_flags, res->current_access_flags,
                res->current_stage_flags, res->current_stage_flags);
            image_barriers.push_back(img_bar);
        }
    }

    return {};
}

TextureHandle VkImageIn::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<VkImageResource>(resource)->tex;
}

std::shared_ptr<VkImageIn> VkImageIn::compute_read(const std::string& name, const uint32_t delay) {
    return std::make_shared<VkImageIn>(
        name, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageUsageFlagBits::eSampled,
        vk::ShaderStageFlagBits::eCompute, delay);
}

std::shared_ptr<VkImageIn> VkImageIn::transfer_src(const std::string& name, const uint32_t delay) {
    return std::make_shared<VkImageIn>(
        name, vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eAllTransfer,
        vk::ImageLayout::eTransferSrcOptimal, vk::ImageUsageFlagBits::eTransferSrc,
        vk::ShaderStageFlags(), delay);
}

} // namespace merian_nodes
