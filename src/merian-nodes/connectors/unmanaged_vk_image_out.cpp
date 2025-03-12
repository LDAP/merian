//
// Created by oschdi on 2/10/25.
//

#include <merian-nodes/connectors/unmanaged_vk_image_out.hpp>
#include <merian-nodes/connectors/vk_image_in.hpp>
#include <merian-nodes/resources/image_array_resource.hpp>
#include <merian/utils/pointer.hpp>

namespace merian_nodes {

UnmanagedVkImageOut::UnmanagedVkImageOut(const std::string& name, const uint32_t image_count)
 : VkImageOut(name, false, image_count) {}


GraphResourceHandle UnmanagedVkImageOut::create_resource(
    [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    const ResourceAllocatorHandle& allocator,
    [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resource_index,
    [[maybe_unused]] const uint32_t ring_size) {

    vk::PipelineStageFlags2 input_pipeline_stages;
    vk::AccessFlags2 input_access_flags;
    vk::ImageLayout first_input_layout = vk::ImageLayout::eUndefined;

    for (const auto& [input_node, input] : inputs) {
        const auto& con_in = debugable_ptr_cast<VkImageIn>(input);
        input_pipeline_stages |= con_in->pipeline_stages;
        input_access_flags |= con_in->access_flags;

        if (first_input_layout == vk::ImageLayout::eUndefined) {
            first_input_layout = con_in->required_layout;
        }
    }

    return std::make_shared<ImageArrayResource>(images, allocator->get_dummy_texture()->get_image(),
                                                  input_pipeline_stages, input_access_flags,
                                                  first_input_layout);
}

Connector::ConnectorStatusFlags UnmanagedVkImageOut::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    auto res = debugable_ptr_cast<ImageArrayResource>(resource);
    if (!res->current_updates.empty()) {
        res->pending_updates.clear();
        std::swap(res->pending_updates, res->current_updates);
        return NEEDS_DESCRIPTOR_UPDATE;
    }

    return {};
}

UnmanagedVkImageOutHandle UnmanagedVkImageOut::create(const std::string& name,
                                                      const uint32_t array_size) {
    return std::make_shared<UnmanagedVkImageOut>(name, array_size);
}


} // namespace merian_nodes