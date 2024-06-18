#include "vk_texture_array_in.hpp"

#include "merian-nodes/graph/errors.hpp"

namespace merian_nodes {

VkTextureArrayIn::VkTextureArrayIn(const std::string& name,
                               const vk::ShaderStageFlags stage_flags,
                               const vk::ImageLayout required_layout,
                               const vk::AccessFlags2 access_flags,
                               const vk::PipelineStageFlags2 pipeline_stages)
    : TypedInputConnector(name, 0), stage_flags(stage_flags), required_layout(required_layout),
      access_flags(access_flags), pipeline_stages(pipeline_stages) {

    assert(required_layout != vk::ImageLayout::eUndefined && access_flags && pipeline_stages);
    assert(!stage_flags || (required_layout == vk::ImageLayout::eShaderReadOnlyOptimal &&
                            access_flags & vk::AccessFlagBits2::eShaderRead));
}

std::optional<vk::DescriptorSetLayoutBinding> VkTextureArrayIn::get_descriptor_info() const {
    if (stage_flags)
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eCombinedImageSampler,
                                              array_size, stage_flags, nullptr};
    return std::nullopt;
}

void VkTextureArrayIn::get_descriptor_update(const uint32_t binding,
                                           GraphResourceHandle& resource,
                                           DescriptorSetUpdate& update) {
    const auto& res = debugable_ptr_cast<TextureArrayResource>(resource);
    for (auto& pending_update : res->pending_updates) {
        const TextureHandle tex =
            res->textures[pending_update] ? res->textures[pending_update] : res->dummy_texture;
        update.write_descriptor_texture(binding, tex, pending_update, 1, required_layout);
    }
}

Connector::ConnectorStatusFlags VkTextureArrayIn::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const vk::CommandBuffer& cmd,
    GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    const auto& res = debugable_ptr_cast<TextureArrayResource>(resource);

    for (auto& tex : res->textures) {
        if (!tex) {
            continue;
        }
        // No barrier required, if no transition required
        if (required_layout != tex->get_image()->get_current_layout()) {
            const vk::ImageMemoryBarrier2 img_bar = tex->get_image()->barrier2(
                required_layout, res->input_access_flags, res->input_access_flags,
                res->input_stage_flags, res->input_stage_flags);
            image_barriers.push_back(img_bar);
        }
    }

    return {};
}

const TextureArrayResource& VkTextureArrayIn::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<const TextureArrayResource>(resource);
}

void VkTextureArrayIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<VkTextureArrayOut>(output);
    if (!casted_output) {
        throw graph_errors::connector_error{
            fmt::format("TextureArrayIn {} cannot recive from {}.", name, output->name)};
    }
    array_size = casted_output->textures.size();
}

VkTextureArrayInHandle VkTextureArrayIn::compute_read(const std::string& name) {
    return std::make_shared<VkTextureArrayIn>(
        name, vk::ShaderStageFlagBits::eCompute, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader);
}

} // namespace merian_nodes
