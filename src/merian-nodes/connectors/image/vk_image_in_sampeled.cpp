#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"

namespace merian {

VkSampledImageIn::VkSampledImageIn(const vk::AccessFlags2 access_flags,
                                   const vk::PipelineStageFlags2 pipeline_stages,
                                   const vk::ImageUsageFlags usage_flags,
                                   const vk::ShaderStageFlags stage_flags,
                                   const uint32_t delay,
                                   const bool optional,
                                   const std::optional<SamplerHandle>& overwrite_sampler,
                                   const vk::ImageLayout required_layout)
    : VkImageIn(access_flags,
                pipeline_stages,
                required_layout,
                usage_flags,
                stage_flags,
                delay,
                optional),
      overwrite_sampler(overwrite_sampler) {
    assert(stage_flags);
}

std::optional<vk::DescriptorSetLayoutBinding> VkSampledImageIn::get_descriptor_info() const {
    return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eCombinedImageSampler,
                                          get_array_size(), get_stage_flags(), nullptr};
}

void VkSampledImageIn::get_descriptor_update(const uint32_t binding,
                                             const GraphResourceHandle& resource,
                                             const DescriptorSetHandle& update,
                                             const ResourceAllocatorHandle& allocator) {
    if (!resource) {
        // the optional connector was not connected
        for (uint32_t i = 0; i < get_array_size(); i++) {
            update->queue_descriptor_write_texture(binding, allocator->get_dummy_texture(), i,
                                                   vk::ImageLayout::eShaderReadOnlyOptimal);
        }

        return;
    }

    const auto& res = debugable_ptr_cast<ImageArrayResource>(resource);

    for (auto& update_idx : res->pending_updates) {
        TextureHandle tex = res->get_texture(update_idx);
        if (tex) {
            if (overwrite_sampler) {
                tex = Texture::create(tex->get_view(), *overwrite_sampler);
            }
            update->queue_descriptor_write_texture(binding, tex, update_idx,
                                                   vk::ImageLayout::eShaderReadOnlyOptimal);
        } else {

            update->queue_descriptor_write_texture(binding, allocator->get_dummy_texture(),
                                                   update_idx,
                                                   vk::ImageLayout::eShaderReadOnlyOptimal);
        }
    }
}

std::shared_ptr<VkSampledImageIn>
VkSampledImageIn::compute_read(const uint32_t delay,
                               const bool optional,
                               const std::optional<SamplerHandle>& overwrite_sampler) {
    return std::make_shared<VkSampledImageIn>(
        vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageUsageFlagBits::eSampled, vk::ShaderStageFlagBits::eCompute, delay, optional,
        overwrite_sampler);
}

std::shared_ptr<VkSampledImageIn> VkSampledImageIn::fragment_read(const uint32_t delay,
                                                                  const bool optional) {
    return std::make_shared<VkSampledImageIn>(
        vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageUsageFlagBits::eSampled, vk::ShaderStageFlagBits::eFragment, delay, optional);
}

} // namespace merian
