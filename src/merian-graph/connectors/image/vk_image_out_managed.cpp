#include "merian-graph/connectors/image/vk_image_out_managed.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/resources/image_array_resource_managed.hpp"

namespace merian {

ManagedVkImageOut::ManagedVkImageOut(const vk::ArrayProxy<vk::ImageCreateInfo>& create_info,
                                     const bool persistent)
    : VkImageOut(persistent, create_info.size()),
      create_infos(create_info.begin(), create_info.end()) {
    assert(!create_info.empty());
}

const ImageArrayResource& ManagedVkImageOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<const ImageArrayResource>(resource);
}

void ManagedVkImageOut::bind(ShaderCursor& cursor,
                             const GraphResourceHandle& resource,
                             [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                             [[maybe_unused]] const ConnectorAccess& access) {
    const auto& res = debugable_ptr_cast<ImageArrayResource>(resource);
    // a shader may only bind an output that was allocated with a view (usage shader-accessible).
    assert(res->get_texture(0) && "shader-bound managed image output has no view");
    if (get_array_size() == 1) {
        cursor.write(res->get_texture(0)->get_view(), vk::ImageLayout::eGeneral);
    } else {
        for (uint32_t i = 0; i < get_array_size(); i++) {
            cursor[i].write(res->get_texture(i)->get_view(), vk::ImageLayout::eGeneral);
        }
    }
}

Connector::ConnectorStatusFlags ManagedVkImageOut::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    constexpr vk::AccessFlags2 all_access =
        vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead;
    constexpr vk::PipelineStageFlags2 all_stages = vk::PipelineStageFlagBits2::eAllCommands;

    const auto& res = debugable_ptr_cast<ManagedImageArrayResource>(resource);
    for (const auto& image : res->images) {
        if (!persistent || image->get_current_layout() != vk::ImageLayout::eGeneral) {
            image_barriers.push_back(
                image->barrier2(vk::ImageLayout::eGeneral, all_access, all_access, all_stages,
                                all_stages, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                all_levels_and_layers(), !persistent));
        }
    }

    return {};
}

GraphResourceHandle ManagedVkImageOut::create_resource(
    [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    const ConnectorAccess& combined_access,
    const ResourceAllocatorHandle& allocator,
    const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resource_index,
    [[maybe_unused]] const uint32_t ring_size) {
    const ResourceAllocatorHandle alloc = persistent ? allocator : aliasing_allocator;

    const auto res = std::make_shared<ManagedImageArrayResource>(get_array_size());

    assert(get_array_size() == create_infos.size());
    for (uint32_t i = 0; i < get_array_size(); i++) {
        vk::ImageCreateInfo create_info = create_infos[i];
        create_info.usage |= combined_access.image_usage;
        res->images[i] = alloc->create_image(create_info, MemoryMappingType::NONE);
    }

    const bool needs_view =
        merian::Image::valid_for_view(combined_access.image_usage | get_create_info()->usage);

    if (needs_view) {
        res->textures.emplace(get_array_size());

        for (uint32_t i = 0; i < get_array_size(); i++) {
            res->textures.value()[i] =
                allocator->create_texture(res->images[i], res->images[i]->make_view_create_info());
        }
    }

    return res;
}

std::optional<vk::ImageCreateInfo> ManagedVkImageOut::get_create_info(const uint32_t index) const {
    assert(index < create_infos.size());
    return create_infos[index];
}

ManagedVkImageOutHandle ManagedVkImageOut::create(const vk::Format format,
                                                  const vk::Extent3D extent,
                                                  const bool persistent) {
    const vk::ImageCreateInfo create_info{
        {},
        extent.depth == 1 ? vk::ImageType::e2D : vk::ImageType::e3D,
        format,
        extent,
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        {},
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };

    return std::make_shared<ManagedVkImageOut>(create_info, persistent);
}

ManagedVkImageOutHandle ManagedVkImageOut::create(const vk::Format format,
                                                  const uint32_t width,
                                                  const uint32_t height,
                                                  const uint32_t depth,
                                                  const bool persistent) {
    return create(format, vk::Extent3D{width, height, depth}, persistent);
}

ManagedVkImageOutHandle
ManagedVkImageOut::create(const vk::ArrayProxy<vk::ImageCreateInfo>& create_info,
                          const bool persistent) {
    return std::make_shared<ManagedVkImageOut>(create_info, persistent);
}

} // namespace merian
