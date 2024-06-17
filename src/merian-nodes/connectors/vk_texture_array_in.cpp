#include "vk_texture_array_in.hpp"

#include "merian-nodes/graph/errors.hpp"

namespace merian_nodes {

TextureArrayIn::TextureArrayIn(const std::string& name,
                               const std::optional<vk::ShaderStageFlags>& stage_flags)
    : TypedInputConnector(name, 0), stage_flags(stage_flags) {}

std::optional<vk::DescriptorSetLayoutBinding> TextureArrayIn::get_descriptor_info() const {
    if (stage_flags)
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eCombinedImageSampler,
                                              array_size, *stage_flags, nullptr};
    return std::nullopt;
}

void TextureArrayIn::get_descriptor_update(const uint32_t binding,
                                           GraphResourceHandle& resource,
                                           DescriptorSetUpdate& update) {
    const auto& res = debugable_ptr_cast<TextureArrayResource>(resource);
    for (auto& pending_update : res->pending_updates) {
        update.write_descriptor_texture(binding, res->textures[pending_update], pending_update, 1,
                                        vk::ImageLayout::eShaderReadOnlyOptimal);
    }
}

const TextureArrayResource& TextureArrayIn::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<const TextureArrayResource>(resource);
}

void TextureArrayIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<TextureArrayOut>(output);
    if (!casted_output) {
        throw graph_errors::connector_error{
            fmt::format("TextureArrayIn {} cannot recive from {}.", name, output->name)};
    }
    array_size = casted_output->textures.size();
}

TextureArrayInHandle
TextureArrayIn::create(const std::string& name,
                       const std::optional<vk::ShaderStageFlags>& stage_flags) {
    return std::make_shared<TextureArrayIn>(name, stage_flags);
}

} // namespace merian_nodes
