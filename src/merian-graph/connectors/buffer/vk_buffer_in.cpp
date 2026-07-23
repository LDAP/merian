#include "merian-graph/connectors/buffer/vk_buffer_in.hpp"
#include "merian-graph/graph/errors.hpp"

namespace merian {

const BufferArrayResource& VkBufferIn::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<const BufferArrayResource>(resource);
}

void VkBufferIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<VkBufferOut>(output);
    if (!casted_output) {
        throw graph_errors::invalid_connection{"VkBufferIn cannot receive from output."};
    }

    array_size = casted_output->get_array_size();
}

void VkBufferIn::bind(ShaderCursor& cursor,
                      const GraphResourceHandle& resource,
                      const ResourceAllocatorHandle& allocator,
                      [[maybe_unused]] const ConnectorAccess& access) {
    const auto write = [&](ShaderCursor field, const uint32_t index) {
        const BufferHandle buffer =
            resource ? debugable_ptr_cast<BufferArrayResource>(resource)->get_buffer(index)
                     : nullptr;
        field.write(buffer ? buffer : allocator->get_dummy_buffer());
    };
    if (get_array_size() == 1) {
        write(cursor, 0);
    } else {
        for (uint32_t i = 0; i < get_array_size(); i++) {
            write(cursor[i], i);
        }
    }
}

VkBufferInHandle VkBufferIn::create() {
    return std::make_shared<VkBufferIn>();
}

} // namespace merian
