#include "merian-graph/connectors/image/vk_image_in.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian-graph/graph/node.hpp"

namespace merian {

Connector::ConnectorStatusFlags
VkImageIn::on_pre_process([[maybe_unused]] GraphRun& run,
                          [[maybe_unused]] const CommandBufferHandle& cmd,
                          const GraphResourceHandle& resource,
                          [[maybe_unused]] const NodeHandle& node,
                          std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                          [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    constexpr vk::AccessFlags2 all_access =
        vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead;
    constexpr vk::PipelineStageFlags2 all_stages = vk::PipelineStageFlagBits2::eAllCommands;

    if (!resource) {
        return {};
    }
    const auto& res = debugable_ptr_cast<ImageArrayResource>(resource);
    for (uint32_t i = 0; i < get_array_size(); i++) {
        const auto& image = res->get_image(i);
        if (image && image->get_current_layout() != vk::ImageLayout::eGeneral) {
            image_barriers.push_back(image->barrier2(vk::ImageLayout::eGeneral, all_access,
                                                     all_access, all_stages, all_stages));
        }
    }
    return {};
}

void VkImageIn::on_connect_output(const OutputConnectorHandle& output) {
    const auto casted_output = std::dynamic_pointer_cast<VkImageOut>(output);

    if (!casted_output) {
        throw graph_errors::invalid_connection{
            "This connector cannot receive from output. Only connectors "
            "derived from VkImageOut are supported."};
    }

    array_size = casted_output->get_array_size();
}

const ImageArrayResource& VkImageIn::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<const ImageArrayResource>(resource);
}

VkImageInHandle VkImageIn::create() {
    return std::make_shared<VkImageIn>();
}

} // namespace merian
