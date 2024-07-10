#include "vk_texture_array_out.hpp"
#include "vk_texture_array_in.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/utils/pointer.hpp"

namespace merian_nodes {

VkTextureArrayOut::VkTextureArrayOut(const std::string& name, const uint32_t array_size)
    : TypedOutputConnector(name, false), textures(array_size) {}

GraphResourceHandle VkTextureArrayOut::create_resource(
    [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    const ResourceAllocatorHandle& allocator,
    [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resoruce_index,
    const uint32_t ring_size) {

    vk::PipelineStageFlags2 input_pipeline_stages;
    vk::AccessFlags2 input_access_flags;
    vk::ImageLayout first_input_layout = vk::ImageLayout::eUndefined;

    for (auto& [input_node, input] : inputs) {
        const auto& con_in = debugable_ptr_cast<VkTextureArrayIn>(input);
        input_pipeline_stages |= con_in->pipeline_stages;
        input_access_flags |= con_in->access_flags;

        if (first_input_layout == vk::ImageLayout::eUndefined) {
            first_input_layout = con_in->required_layout;
        }
    }

    return std::make_shared<TextureArrayResource>(
        textures, ring_size, allocator->get_dummy_texture(), input_pipeline_stages,
        input_access_flags, first_input_layout);
}

TextureArrayResource& VkTextureArrayOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<TextureArrayResource>(resource);
}

Connector::ConnectorStatusFlags VkTextureArrayOut::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const vk::CommandBuffer& cmd,
    const GraphResourceHandle& resource,
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

Connector::ConnectorStatusFlags VkTextureArrayOut::on_post_process(
    GraphRun& run,
    [[maybe_unused]] const vk::CommandBuffer& cmd,
    const GraphResourceHandle& resource,
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

    res->in_flight_textures[run.get_in_flight_index()] = textures;

    return flags;
}

VkTextureArrayOutHandle VkTextureArrayOut::create(const std::string& name,
                                                  const uint32_t array_size) {
    return std::make_shared<VkTextureArrayOut>(name, array_size);
}

uint32_t VkTextureArrayOut::array_size() const {
    return textures.size();
}

} // namespace merian_nodes
