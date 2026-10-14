#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"

#include "merian-graph/resources/buffer_array_resource_managed.hpp"
#include "merian/utils/pointer.hpp"

namespace merian {

ManagedVkBufferOut::ManagedVkBufferOut(const vk::BufferCreateInfo& create_info,
                                       const uint32_t array_size)
    : VkBufferOut(array_size), create_info(create_info) {}

GraphResourceHandle ManagedVkBufferOut::create_resource(const ResourceCreateInfo& info) {
    vk::BufferCreateInfo buffer_create_info = create_info;
    buffer_create_info.usage |= info.combined_access.buffer_usage;

    const auto res = std::make_shared<ManagedBufferArrayResource>(get_array_size());

    for (uint32_t i = 0; i < get_array_size(); i++) {
        res->buffers[i] = info.allocator->create_buffer(
            buffer_create_info, MemoryMappingType::NONE,
            get_array_size() == 1 ? info.debug_name : fmt::format("{}[{}]", info.debug_name, i));
    }

    return res;
}

void ManagedVkBufferOut::bind(ShaderCursor& cursor,
                              const GraphResourceHandle& resource,
                              const ResourceAllocatorHandle& allocator,
                              [[maybe_unused]] const ConnectorAccess& access) {
    const auto write = [&](ShaderCursor field, const uint32_t index) {
        field.write(resource ? debugable_ptr_cast<BufferArrayResource>(resource)->get_buffer(index)
                             : allocator->get_dummy_buffer());
    };
    if (get_array_size() == 1) {
        write(cursor, 0);
    } else {
        for (uint32_t i = 0; i < get_array_size(); i++) {
            write(cursor[i], i);
        }
    }
}

BufferArrayResource& ManagedVkBufferOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<BufferArrayResource>(resource);
}

ManagedVkBufferOutHandle ManagedVkBufferOut::create(const vk::BufferCreateInfo& create_info,
                                                    const uint32_t array_size) {
    return std::make_shared<ManagedVkBufferOut>(create_info, array_size);
}

} // namespace merian
