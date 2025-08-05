#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"

namespace merian_nodes {

VkSampledImageIn::VkSampledImageIn(const std::string& name,
                                   const vk::AccessFlags2 access_flags,
                                   const vk::PipelineStageFlags2 pipeline_stages,
                                   const vk::ImageUsageFlags usage_flags,
                                   const vk::ShaderStageFlags stage_flags,
                                   const uint32_t delay,
                                   const bool optional,
                                   const std::optional<SamplerHandle>& overwrite_sampler,
                                   const vk::ImageLayout required_layout)
    : VkImageIn(name,
                access_flags,
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
    assert(res->textures.has_value());

    for (auto& update_idx : res->pending_updates) {
        TextureHandle tex;
        if (overwrite_sampler) {
            tex =
                Texture::create(res->textures.value()[update_idx]->get_view(), *overwrite_sampler);
        } else {
            tex = res->textures.value()[update_idx];
            assert(tex);
        }

        update->queue_descriptor_write_texture(binding, tex, update_idx,
                                               vk::ImageLayout::eShaderReadOnlyOptimal);
    }
}

std::shared_ptr<VkSampledImageIn>
VkSampledImageIn::compute_read(const std::string& name, const uint32_t delay, const bool optional) {
    return std::make_shared<VkSampledImageIn>(
        name, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageUsageFlagBits::eSampled, vk::ShaderStageFlagBits::eCompute, delay, optional);
}

std::shared_ptr<VkSampledImageIn> VkSampledImageIn::fragment_read(const std::string& name,
                                                                  const uint32_t delay,
                                                                  const bool optional) {
    return std::make_shared<VkSampledImageIn>(
        name, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageUsageFlagBits::eSampled, vk::ShaderStageFlagBits::eFragment, delay, optional);
}

} // namespace merian_nodes
