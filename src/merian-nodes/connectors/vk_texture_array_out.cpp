#include "vk_texture_array_out.hpp"
#include "graph/node.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian/utils/pointer.hpp"

namespace merian_nodes {

TextureArrayOut::TextureArrayOut(const std::string& name,
                                 const uint32_t array_size,
                                 const std::optional<vk::ShaderStageFlags>& stage_flags)
    : TypedOutputConnector(name, false), array_size(array_size), stage_flags(stage_flags) {}

std::optional<vk::DescriptorSetLayoutBinding> TextureArrayOut::get_descriptor_info() const {
    if (stage_flags)
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eCombinedImageSampler,
                                              array_size, *stage_flags, nullptr};
    return std::nullopt;
}

void TextureArrayOut::get_descriptor_update([[maybe_unused]] const uint32_t binding,
                                            [[maybe_unused]] GraphResourceHandle& resource,
                                            [[maybe_unused]] DescriptorSetUpdate& update) {
    const auto& res = debugable_ptr_cast<TextureArrayResource>(resource);
    for (auto& pending_update : res->pending_updates) {
        update.write_descriptor_texture(binding, res->textures[pending_update], pending_update, 1,
                                        vk::ImageLayout::eShaderReadOnlyOptimal);
    }
}

GraphResourceHandle TextureArrayOut::create_resource(
    [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    [[maybe_unused]] const ResourceAllocatorHandle& allocator,
    [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resoruce_index,
    const uint32_t ring_size) {

    return std::make_shared<TextureArrayResource>(array_size, ring_size);
}

TextureArrayResource& TextureArrayOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<TextureArrayResource>(resource);
}

Connector::ConnectorStatusFlags TextureArrayOut::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const vk::CommandBuffer& cmd,
    GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<TextureArrayResource>(resource);
    if (!res->current_updates.empty()) {
        res->pending_updates.clear();
        std::swap(res->pending_updates, res->current_updates);
        return NEEDS_DESCRIPTOR_UPDATE;
    }

    return {};
}

Connector::ConnectorStatusFlags TextureArrayOut::on_post_process(
    GraphRun& run,
    [[maybe_unused]] const vk::CommandBuffer& cmd,
    GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<TextureArrayResource>(resource);

    Connector::ConnectorStatusFlags flags{};
    if (!res->current_updates.empty()) {
        res->pending_updates.clear();
        std::swap(res->pending_updates, res->current_updates);
        flags |= NEEDS_DESCRIPTOR_UPDATE;
    }

    for (uint32_t i = 0; i < array_size; i++) {
        if (!res->textures[i]) {
            throw graph_errors::connector_error{fmt::format(
                "array index {} if connector {} in node {} was not set.", i, name, node->name)};
        }
        res->in_flight_textures[run.get_in_flight_index()][i] = res->textures[i];
        if (res->textures[i]->get_current_layout() != vk::ImageLayout::eShaderReadOnlyOptimal) {
            image_barriers.emplace_back(
                res->textures[i]->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
        }
    }

    return flags;
}

TextureArrayOutHandle TextureArrayOut::create(const std::string& name, const uint32_t array_size) {
    return std::make_shared<TextureArrayOut>(name, array_size);
}

} // namespace merian_nodes
