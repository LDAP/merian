#include "merian-graph/nodes/host_source/host_image_source.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"

#include <cstring>

namespace merian {

void HostImageSource::initialize(const ContextHandle& context,
                                 const ResourceAllocatorHandle& allocator) {
    HostSource::initialize(context, allocator);
    const vk::Format initial_format = vk::Format::eR32G32B32A32Sfloat;
    upload(initial_format, {1, 1, 1},
           [&](void* texels) { std::memset(texels, 0, Image::format_size(initial_format)); });
}

std::vector<OutputConnectorDescriptor>
HostImageSource::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    create_output([&] { con_out = ManagedVkImageOut::create(format, extent); });
    return {{.name = "out",
             .connector = con_out,
             .access = ConnectorAccess::transfer_dst,
             .persistent = true}};
}

void HostImageSource::upload(const vk::Format format,
                             const vk::Extent3D extent,
                             const std::function<void(void* texels)>& fill) {
    stage_upload(Image::format_size(format) * extent.width * extent.height * extent.depth, fill,
                 [&] {
                     const bool changed = format != this->format || extent != this->extent;
                     this->format = format;
                     this->extent = extent;
                     return changed;
                 });
}

void HostImageSource::cmd_copy(const CommandBufferHandle& cmd,
                               const NodeIO& io,
                               const BufferHandle& staging_buffer,
                               const vk::DeviceSize staging_offset) {
    cmd->copy(staging_buffer, io[con_out],
              vk::BufferImageCopy{staging_offset, 0, 0, first_layer(), {}, extent});
}

} // namespace merian
