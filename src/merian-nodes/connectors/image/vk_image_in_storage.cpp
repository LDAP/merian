#include "merian-nodes/connectors/image/vk_image_in_storage.hpp"

namespace merian_nodes {

VkStorageImageIn::VkStorageImageIn(const std::string& name,
                                   const vk::AccessFlags2 access_flags,
                                   const vk::PipelineStageFlags2 pipeline_stages,
                                   const vk::ImageUsageFlags usage_flags,
                                   const vk::ShaderStageFlags stage_flags,
                                   const uint32_t delay,
                                   const bool optional,
                                   const vk::ImageLayout required_layout)
    : VkImageIn(name,
                access_flags,
                pipeline_stages,
                required_layout,
                usage_flags,
                stage_flags,
                delay,
                optional) {
    assert(stage_flags);
}

std::optional<vk::DescriptorSetLayoutBinding> VkStorageImageIn::get_descriptor_info() const {
    return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageImage, get_array_size(),
                                          get_stage_flags(), nullptr};
}

void VkStorageImageIn::get_descriptor_update(const uint32_t binding,
                                             const GraphResourceHandle& resource,
                                             const DescriptorSetHandle& update,
                                             const ResourceAllocatorHandle& allocator) {
    if (!resource) {
        // the optional connector was not connected
        for (uint32_t i = 0; i < get_array_size(); i++) {
            update->queue_descriptor_write_image(binding, allocator->get_dummy_storage_image_view(),
                                                 i, vk::ImageLayout::eGeneral);
        }

        return;
    }

    const auto& res = debugable_ptr_cast<ImageArrayResource>(resource);
    assert(res->textures.has_value());

    for (auto& update_idx : res->pending_updates) {
        const ImageViewHandle view = res->textures.value()[update_idx]->get_view();
        update->queue_descriptor_write_image(binding, view, update_idx, vk::ImageLayout::eGeneral);
    }
}

std::shared_ptr<VkStorageImageIn>
VkStorageImageIn::compute_read(const std::string& name, const uint32_t delay, const bool optional) {
    return std::make_shared<VkStorageImageIn>(
        name, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageUsageFlagBits::eStorage,
        vk::ShaderStageFlagBits::eCompute, delay, optional);
}

} // namespace merian_nodes
