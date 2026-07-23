#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"

#include "merian-graph/resources/buffer_array_resource_managed.hpp"
#include "merian/utils/pointer.hpp"

namespace merian {

ManagedVkBufferOut::ManagedVkBufferOut(const vk::BufferCreateInfo& create_info,
                                       const bool persistent,
                                       const uint32_t array_size)
    : VkBufferOut(!persistent, array_size), create_info(create_info), persistent(persistent) {}

GraphResourceHandle ManagedVkBufferOut::create_resource(
    [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    const ConnectorAccess& combined_access,
    const ResourceAllocatorHandle& allocator,
    const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resource_index,
    [[maybe_unused]] const uint32_t ring_size) {
    vk::BufferCreateInfo buffer_create_info = create_info;
    buffer_create_info.usage |= combined_access.buffer_usage;

    const ResourceAllocatorHandle alloc = persistent ? allocator : aliasing_allocator;

    const auto res = std::make_shared<ManagedBufferArrayResource>(get_array_size());

    for (uint32_t i = 0; i < get_array_size(); i++) {
        res->buffers[i] = alloc->create_buffer(buffer_create_info, MemoryMappingType::NONE);
    }

    return res;
}

void ManagedVkBufferOut::bind(ShaderCursor& cursor,
                              const GraphResourceHandle& resource,
                              [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                              [[maybe_unused]] const ConnectorAccess& access) {
    const auto& res = debugable_ptr_cast<BufferArrayResource>(resource);
    if (get_array_size() == 1) {
        cursor.write(res->get_buffer(0));
    } else {
        for (uint32_t i = 0; i < get_array_size(); i++) {
            cursor[i].write(res->get_buffer(i));
        }
    }
}

BufferArrayResource& ManagedVkBufferOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<BufferArrayResource>(resource);
}

ManagedVkBufferOutHandle ManagedVkBufferOut::create(const vk::BufferCreateInfo& create_info,
                                                    const bool persistent,
                                                    const uint32_t array_size) {
    return std::make_shared<ManagedVkBufferOut>(create_info, persistent, array_size);
}

} // namespace merian
