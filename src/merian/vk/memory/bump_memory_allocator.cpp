#include "merian/vk/memory/bump_memory_allocator.hpp"

#include <bit>
#include <limits>

namespace merian {

BumpMemoryAllocation::BumpMemoryAllocation(const ContextHandle& context,
                                           const BumpMemoryAllocatorHandle& allocator,
                                           const vk::DeviceSize offset,
                                           const vk::DeviceSize size)
    : MemoryAllocation(context), allocator(allocator), offset(offset), size(size) {
    SPDLOG_TRACE("create bump suballocation ({})", fmt::ptr(this));
}

BumpMemoryAllocation::~BumpMemoryAllocation() {
    SPDLOG_TRACE("free bump suballocation ({})", fmt::ptr(this));
}

void BumpMemoryAllocation::invalidate(const VkDeviceSize offset, const VkDeviceSize size) {
    allocator->get_base_buffer()->get_memory()->invalidate(
        this->offset + offset, size == VK_WHOLE_SIZE ? this->size : size);
}

void BumpMemoryAllocation::flush(const VkDeviceSize offset, const VkDeviceSize size) {
    allocator->get_base_buffer()->get_memory()->flush(this->offset + offset,
                                                      size == VK_WHOLE_SIZE ? this->size : size);
}

void* BumpMemoryAllocation::map() {
    void* const base = allocator->get_mapped_base();
    if (base == nullptr) {
        throw std::runtime_error{
            "mapping is unsupported for this BumpMemoryAllocator (non host-visible buffer)"};
    }
    return static_cast<int8_t*>(base) + offset;
}

void BumpMemoryAllocation::unmap() {}

MemoryAllocationInfo BumpMemoryAllocation::get_memory_info() const {
    const MemoryAllocationInfo& base = allocator->get_base_buffer()->get_memory()->get_memory_info();
    return MemoryAllocationInfo{
        base.memory,
        base.offset + offset,
        size,
        base.memory_type_index,
        name.empty() ? nullptr : name.c_str(),
    };
}

ImageHandle
BumpMemoryAllocation::create_aliasing_image(const vk::ImageCreateInfo& image_create_info,
                                            const vk::DeviceSize allocation_offset) {
    const ImageHandle image = allocator->get_base_buffer()->get_memory()->create_aliasing_image(
        image_create_info, offset + allocation_offset);
    image->_set_memory_allocation(shared_from_this());
    return image;
}

BufferHandle
BumpMemoryAllocation::create_aliasing_buffer(const vk::BufferCreateInfo& buffer_create_info,
                                             const vk::DeviceSize allocation_offset) {
    const BufferHandle buffer = allocator->get_base_buffer()->get_memory()->create_aliasing_buffer(
        buffer_create_info, offset + allocation_offset);
    buffer->_set_memory_allocation(shared_from_this());
    return buffer;
}

void BumpMemoryAllocation::bind_to_image(const ImageHandle& image,
                                         const vk::DeviceSize allocation_offset) {
    allocator->get_base_buffer()->get_memory()->bind_to_image(image, offset + allocation_offset);
    image->_set_memory_allocation(shared_from_this());
}

void BumpMemoryAllocation::bind_to_buffer(const BufferHandle& buffer,
                                          const vk::DeviceSize allocation_offset) {
    allocator->get_base_buffer()->get_memory()->bind_to_buffer(buffer, offset + allocation_offset);
    buffer->_set_memory_allocation(shared_from_this());
}

MemoryAllocatorHandle BumpMemoryAllocation::get_allocator() const {
    return allocator;
}

const BumpMemoryAllocatorHandle& BumpMemoryAllocation::get_bump_allocator() const {
    return allocator;
}

const vk::DeviceSize& BumpMemoryAllocation::get_size() const {
    return size;
}

const vk::DeviceSize& BumpMemoryAllocation::get_offset() const {
    return offset;
}

void BumpMemoryAllocation::properties(Properties& props) {
    MemoryAllocation::properties(props);

    if (props.st_begin_child("suballocation from buffer")) {
        allocator->get_base_buffer()->properties(props);
        props.st_end_child();
    }
}

// ------------------------------------------------------------------------------------

BumpMemoryAllocator::BumpMemoryAllocator(const BufferHandle& buffer)
    : MemoryAllocator(buffer->get_context()), buffer(buffer),
      buffer_info(buffer->get_memory()->get_memory_info()) {

    buffer_flags = buffer->get_context()
                       ->get_physical_device()
                       ->get_memory_properties()
                       .memoryProperties.memoryTypes[buffer_info.memory_type_index]
                       .propertyFlags;
    buffer_alignment =
        vk::DeviceSize(1)
        << std::min(std::countr_zero(buffer_info.offset),
                    std::numeric_limits<vk::DeviceSize>::digits - 1);

    if (buffer_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        mapped_base = buffer->get_memory()->map();
    }
}

BumpMemoryAllocator::~BumpMemoryAllocator() {
    if (mapped_base != nullptr) {
        buffer->get_memory()->unmap();
    }
}

vk::DeviceSize BumpMemoryAllocator::allocate(const vk::MemoryRequirements& requirements) {
    assert(requirements.alignment == 0ul || std::popcount(requirements.alignment) == 1);

    const vk::DeviceSize alignment = std::max(requirements.alignment, vk::DeviceSize(1));
    const vk::DeviceSize total = buffer_info.size;

    vk::DeviceSize cur = current_offset.load(std::memory_order_relaxed);
    while (true) {
        // Align so the device address (buffer_info.offset + offset) satisfies `alignment`. When
        // alignment <= buffer_alignment this is equivalent to buffer-relative alignment.
        const vk::DeviceSize offset =
            ((buffer_info.offset + cur + alignment - 1ul) & -alignment) - buffer_info.offset;
        const vk::DeviceSize end = offset + requirements.size;
        if (end > total) {
            throw AllocationFailed(vk::Result::eErrorOutOfDeviceMemory,
                                   "BumpMemoryAllocator: out of space");
        }
        if (current_offset.compare_exchange_weak(cur, end, std::memory_order_release,
                                                 std::memory_order_relaxed)) {
            return offset;
        }
    }
}

MemoryAllocationHandle
BumpMemoryAllocator::allocate_memory(const vk::MemoryPropertyFlags required_flags,
                                     const vk::MemoryRequirements& requirements,
                                     [[maybe_unused]] const std::string& debug_name,
                                     const MemoryMappingType mapping_type,
                                     const vk::MemoryPropertyFlags /* preferred_flags */,
                                     const bool dedicated,
                                     const float /* dedicated_priority */) {
    if (dedicated) {
        SPDLOG_WARN("requested a dedicated memory allocation. But this is a bump allocator which "
                    "cannot allocate dedicated memory.");
    }

    if ((required_flags & buffer_flags) != required_flags) {
        throw AllocationFailed(vk::Result::eErrorOutOfDeviceMemory,
                               fmt::format("BumpMemoryAllocator: buffer has properties {}, but "
                                           "required was {}",
                                           vk::to_string(buffer_flags),
                                           vk::to_string(required_flags)));
    }

    if (mapping_type != MemoryMappingType::NONE &&
        !(buffer_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        throw AllocationFailed(
            vk::Result::eErrorOutOfDeviceMemory,
            "BumpMemoryAllocator: mappable allocation requested but buffer is not host visible");
    }

    if ((requirements.memoryTypeBits & (1u << buffer_info.memory_type_index)) == 0u) {
        throw AllocationFailed(
            vk::Result::eErrorOutOfDeviceMemory,
            fmt::format("BumpMemoryAllocator: buffer memory type index {} not supported by "
                        "the requested allocation",
                        buffer_info.memory_type_index));
    }

    const vk::DeviceSize offset = allocate(requirements);

    const auto self = std::static_pointer_cast<BumpMemoryAllocator>(shared_from_this());
    const auto allocation =
        std::make_shared<BumpMemoryAllocation>(get_context(), self, offset, requirements.size);

#ifndef NDEBUG
    allocation->name = debug_name;
#endif

    return allocation;
}

const BufferHandle& BumpMemoryAllocator::get_base_buffer() const {
    return buffer;
}

void* BumpMemoryAllocator::get_mapped_base() const {
    return mapped_base;
}

vk::DeviceSize BumpMemoryAllocator::get_free_size() const {
    const vk::DeviceSize cur = current_offset.load(std::memory_order_relaxed);
    return cur >= buffer_info.size ? 0ul : buffer_info.size - cur;
}

BumpMemoryAllocatorHandle BumpMemoryAllocator::create(const BufferHandle& buffer) {
    return BumpMemoryAllocatorHandle(new BumpMemoryAllocator(buffer));
}

} // namespace merian
