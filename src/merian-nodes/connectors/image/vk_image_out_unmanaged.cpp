#include "merian-nodes/connectors/image/vk_image_out_unmanaged.hpp"
#include "merian-nodes/connectors/image/vk_image_in.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/utils/pointer.hpp"

namespace merian_nodes {

UnmanagedVkImageOut::UnmanagedVkImageOut(const std::string& name,
                                         const uint32_t array_size,
                                         const vk::ImageUsageFlags image_usage_flags)
    : VkImageOut(name, false, array_size), image_usage_flags(image_usage_flags) {}

GraphResourceHandle UnmanagedVkImageOut::create_resource(
    [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    const ResourceAllocatorHandle& /*allocator*/,
    [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resource_index,
    [[maybe_unused]] const uint32_t ring_size) {
    assert(resource_index == 0);

    vk::ImageUsageFlags all_usage_flags = image_usage_flags;
    vk::PipelineStageFlags2 input_pipeline_stages;
    vk::AccessFlags2 input_access_flags;
    vk::ImageLayout first_input_layout = vk::ImageLayout::eUndefined;

    for (const auto& [input_node, input] : inputs) {
        const auto& con_in = debugable_ptr_cast<VkImageIn>(input);
        all_usage_flags |= con_in->get_usage_flags();
        input_pipeline_stages |= con_in->get_pipeline_stages();
        input_access_flags |= con_in->get_access_flags();

        if (first_input_layout == vk::ImageLayout::eUndefined) {
            first_input_layout = con_in->get_required_layout();
        }
    }

    if (images.empty()) {
        images.resize(get_array_size());

        if (merian::Image::valid_for_view(all_usage_flags)) {
            textures.emplace(get_array_size());
        }
    } else {
        for (const auto& image : images) {
            if (image && (image->get_usage_flags() & all_usage_flags) != all_usage_flags) {
                throw graph_errors::invalid_connection{
                    fmt::format("images set for the unmanaged output connector {} are missing some "
                                "usage flags for the new inputs.",
                                name)};
            }
        }
    }

    return std::make_shared<UnmanagedImageArrayResource>(images, textures, all_usage_flags,
                                                         input_pipeline_stages, input_access_flags,
                                                         first_input_layout);
}

UnmanagedImageArrayResource& UnmanagedVkImageOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<UnmanagedImageArrayResource>(resource);
}

Connector::ConnectorStatusFlags UnmanagedVkImageOut::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<ImageArrayResource>(resource);
    if (!res->current_updates.empty()) {
        res->pending_updates.clear();
        std::swap(res->pending_updates, res->current_updates);
        return NEEDS_DESCRIPTOR_UPDATE;
    }

    return {};
}

UnmanagedVkImageOutHandle UnmanagedVkImageOut::create(const std::string& name,
                                                      const uint32_t array_size,
                                                      const vk::ImageUsageFlags image_usage_flags) {
    return std::make_shared<UnmanagedVkImageOut>(name, array_size, image_usage_flags);
}

} // namespace merian_nodes
