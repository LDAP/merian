#include "merian/vk/memory/memory_suballocator_vma.hpp"

namespace {

void check_mapping_type(const merian::MemoryMappingType mapping_type,
                        const vk::MemoryPropertyFlags buffer_flags) {
    if (mapping_type != merian::MemoryMappingType::NONE) {
        if (!(buffer_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
            throw merian::AllocationFailed(
                vk::Result::eErrorOutOfDeviceMemory,
                fmt::format("Suballocation failed, this suballocator uses memory which is not host "
                            "visible, but a mappable allocation was requested."));
        }

        if (mapping_type == merian::MemoryMappingType::HOST_ACCESS_RANDOM &&
            !(buffer_flags & vk::MemoryPropertyFlagBits::eHostCached)) {
            SPDLOG_WARN("This suballocator uses memory which is not host "
                        "cached, but a HOST_ACCESS_RANDOM allocation was requested.");
        }
    }
}

} // namespace

namespace merian {

VMAMemorySubAllocation::VMAMemorySubAllocation(
    const ContextHandle& context,
    const std::shared_ptr<VMAMemorySubAllocator>& allocator,
    VmaVirtualAllocation allocation,
    const vk::DeviceSize offset,
    const vk::DeviceSize size)
    : MemoryAllocation(context), allocator(allocator), allocation(allocation), offset(offset),
      size(size) {
    SPDLOG_TRACE("create VMA suballocation ({})", fmt::ptr(this));
}

// frees the memory when called
VMAMemorySubAllocation::~VMAMemorySubAllocation() {
    SPDLOG_TRACE("free VMA suballocation ({})", fmt::ptr(this));
    vmaVirtualFree(allocator->get_vma_block(), allocation);
}

// ------------------------------------------------------------------------------------

void VMAMemorySubAllocation::invalidate(const VkDeviceSize offset, const VkDeviceSize size) {
    allocator->get_base_buffer()->get_memory()->invalidate(
        this->offset + offset, size == VK_WHOLE_SIZE ? this->size : size);
}

void VMAMemorySubAllocation::flush(const VkDeviceSize offset, const VkDeviceSize size) {
    allocator->get_base_buffer()->get_memory()->flush(this->offset + offset,
                                                      size == VK_WHOLE_SIZE ? this->size : size);
}

// Returns a mapping to the suballocation. The offset is already accounted for.
void* VMAMemorySubAllocation::map() {
    return (int8_t*)allocator->get_base_buffer()->get_memory()->map() + offset;
}

void VMAMemorySubAllocation::unmap() {
    allocator->get_base_buffer()->get_memory()->unmap();
}

// Retrieve detailed information about 'memHandle'
// You should not call this to often
MemoryAllocationInfo VMAMemorySubAllocation::get_memory_info() const {
    return MemoryAllocationInfo{
        allocator->buffer_info.memory,
        allocator->buffer_info.offset + offset,
        size,
        allocator->buffer_info.memory_type_index,
        name.empty() ? nullptr : name.c_str(),
    };
}

// ------------------------------------------------------------------------------------

ImageHandle
VMAMemorySubAllocation::create_aliasing_image(const vk::ImageCreateInfo& image_create_info,
                                              const vk::DeviceSize allocation_offset) {

    const ImageHandle image = allocator->get_base_buffer()->get_memory()->create_aliasing_image(
        image_create_info, offset + allocation_offset);
    image->_set_memory_allocation(shared_from_this());
    return image;
}

BufferHandle
VMAMemorySubAllocation::create_aliasing_buffer(const vk::BufferCreateInfo& buffer_create_info,
                                               const vk::DeviceSize allocation_offset) {
    const BufferHandle buffer = allocator->get_base_buffer()->get_memory()->create_aliasing_buffer(
        buffer_create_info, offset + allocation_offset);
    buffer->_set_memory_allocation(shared_from_this());
    return buffer;
}

void VMAMemorySubAllocation::bind_to_image(const ImageHandle& image,
                                           const vk::DeviceSize allocation_offset) {
    allocator->get_base_buffer()->get_memory()->bind_to_image(image, offset + allocation_offset);
    image->_set_memory_allocation(shared_from_this());
}

void VMAMemorySubAllocation::bind_to_buffer(const BufferHandle& buffer,
                                            const vk::DeviceSize allocation_offset) {
    allocator->get_base_buffer()->get_memory()->bind_to_buffer(buffer, offset + allocation_offset);
    buffer->_set_memory_allocation(shared_from_this());
}

// ------------------------------------------------------------------------------------

MemoryAllocatorHandle VMAMemorySubAllocation::get_allocator() const {
    return allocator;
}

const VMAMemorySubAllocatorHandle& VMAMemorySubAllocation::get_suballocator() const {
    return allocator;
}

const vk::DeviceSize& VMAMemorySubAllocation::get_size() const {
    return size;
}

const vk::DeviceSize& VMAMemorySubAllocation::get_offset() const {
    return offset;
}

void VMAMemorySubAllocation::properties(Properties& props) {
    MemoryAllocation::properties(props);

    if (props.st_begin_child("suballocation from buffer")) {
        allocator->get_base_buffer()->properties(props);
        props.st_end_child();
    }
}

// ------------------------------------------------------------------------------------
// Allocator
// ------------------------------------------------------------------------------------

VMAMemorySubAllocator::VMAMemorySubAllocator(const BufferHandle& buffer)
    : MemoryAllocator(buffer->get_context()), buffer(buffer),
      buffer_info(buffer->get_memory()->get_memory_info()) {

    VmaVirtualBlockCreateInfo block_create_info = {};
    block_create_info.size = buffer_info.size;

    VulkanException::throw_if_no_success(vmaCreateVirtualBlock(&block_create_info, &block));

    buffer_flags = buffer->get_context()
                       ->physical_device.physical_device_memory_properties.memoryProperties
                       .memoryTypes[buffer_info.memory_type_index]
                       .propertyFlags;
    buffer_alignment = 1ul << std::countr_zero(buffer_info.offset);
}

VMAMemorySubAllocator::~VMAMemorySubAllocator() {
    vmaDestroyVirtualBlock(block);
}

// ------------------------------------------------------------------------------------

MemoryAllocationHandle
VMAMemorySubAllocator::allocate_memory(const vk::MemoryPropertyFlags required_flags,
                                       const vk::MemoryRequirements& requirements,
                                       [[maybe_unused]] const std::string& debug_name,
                                       const MemoryMappingType mapping_type,
                                       const vk::MemoryPropertyFlags /* preferred_flags */,
                                       const bool dedicated,
                                       const float /* dedicated_priority */) {

    if (dedicated) {
        SPDLOG_WARN("requested a dedicated memory allocation. But this is a suballocator which "
                    "cannot allocate dedicated memory.");
    }

    if ((required_flags & buffer_flags) != required_flags) {
        throw AllocationFailed(vk::Result::eErrorOutOfDeviceMemory,
                               fmt::format("Suballocation failed, this suballocator uses memory "
                                           "with properties {}, but required was {}",
                                           vk::to_string(buffer_flags),
                                           vk::to_string(required_flags)));
    }

    check_mapping_type(mapping_type, buffer_flags);

    if ((requirements.memoryTypeBits & (1u << buffer_info.memory_type_index)) == 0u) {
        throw AllocationFailed(
            vk::Result::eErrorOutOfDeviceMemory,
            fmt::format("Suballocation failed, this suballocator uses memory of type index {}, but "
                        "this type is not supported for the requested allocation",
                        buffer_info.memory_type_index));
    }

    assert(requirements.alignment == 0ul || std::popcount(requirements.alignment) == 0);

    VmaVirtualAllocationCreateInfo alloc_create_info = {};
    if (requirements.alignment <= buffer_alignment) {
        alloc_create_info.size = requirements.size;
        alloc_create_info.alignment = requirements.alignment;
    } else {
        alloc_create_info.size = requirements.size + requirements.alignment;
        alloc_create_info.alignment = 0;
    }

    VmaVirtualAllocation virtual_allocation;
    vk::DeviceSize offset;
    AllocationFailed::throw_if_no_success(
        vmaVirtualAllocate(block, &alloc_create_info, &virtual_allocation, &offset));

    if (requirements.alignment > buffer_alignment) {
        offset = ((buffer_info.offset + offset - 1ul + requirements.alignment) &
                  -requirements.alignment) -
                 buffer_info.offset;
    }

    assert(offset >= buffer_info.offset &&
           offset + requirements.size < buffer_info.offset + buffer_info.size);

    const std::shared_ptr<VMAMemorySubAllocator> allocator =
        static_pointer_cast<VMAMemorySubAllocator>(shared_from_this());

    const auto allocation = std::make_shared<VMAMemorySubAllocation>(
        get_context(), allocator, virtual_allocation, offset, requirements.size);

#ifndef NDEBUG
    allocation->name = debug_name;
#endif

    return allocation;
}

// ------------------------------------------------------------------------------------

const BufferHandle& VMAMemorySubAllocator::get_base_buffer() const {
    return buffer;
}

const VmaVirtualBlock& VMAMemorySubAllocator::get_vma_block() const {
    return block;
}

std::shared_ptr<VMAMemorySubAllocator> VMAMemorySubAllocator::create(const BufferHandle& buffer) {
    const std::shared_ptr<VMAMemorySubAllocator> allocator =
        std::shared_ptr<VMAMemorySubAllocator>(new VMAMemorySubAllocator(buffer));

    return allocator;
}

} // namespace merian
