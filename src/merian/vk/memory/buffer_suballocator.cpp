#include "merian/vk/memory/buffer_suballocator.hpp"

namespace merian {

//////////////////////////////////////////////////////////////////////////

void BufferSubAllocator::init(MemoryAllocator* memAllocator,
                              vk::DeviceSize blockSize,
                              vk::BufferUsageFlags bufferUsageFlags,
                              vk::MemoryPropertyFlags memPropFlags,
                              bool mapped,
                              const std::vector<uint32_t>& sharingQueueFamilyIndices) {
    assert(!m_device);
    m_memAllocator = memAllocator;
    m_device = memAllocator->get_context()->device;
    m_blockSize =
        std::min(blockSize, ((uint64_t(1) << Handle::BLOCKBITS) - 1) * uint64_t(BASE_ALIGNMENT));
    m_bufferUsageFlags = bufferUsageFlags;
    m_memoryPropFlags = memPropFlags;
    m_memoryTypeIndex = ~0;
    m_keepLastBlock = true;
    m_mapped = mapped;
    m_sharingQueueFamilyIndices = sharingQueueFamilyIndices;

    m_freeBlockIndex = INVALID_ID_INDEX;
    m_usedSize = 0;
    m_allocatedSize = 0;
}

void BufferSubAllocator::deinit() {
    if (!m_memAllocator)
        return;

    free(false);

    m_blocks.clear();
    m_memAllocator = nullptr;
}

BufferSubAllocator::Handle BufferSubAllocator::subAllocate(vk::DeviceSize size, uint32_t align) {
    uint32_t usedOffset;
    uint32_t usedSize;
    uint32_t usedAligned;

    uint32_t blockIndex = INVALID_ID_INDEX;

    // if size either doesn't fit in the bits within the handle
    // or we are bigger than the default block size, we use a full dedicated block
    // for this allocation
    bool isDedicated = Handle::needsDedicated(size, align) || size > m_blockSize;

    if (!isDedicated) {
        // Find the first non-dedicated block that can fit the allocation
        for (uint32_t i = 0; i < (uint32_t)m_blocks.size(); i++) {
            Block& block = m_blocks[i];
            if (!block.isDedicated && block.buffer &&
                block.range.subAllocate((uint32_t)size, align, usedOffset, usedAligned, usedSize)) {
                blockIndex = block.index;
                break;
            }
        }
    }

    if (blockIndex == INVALID_ID_INDEX) {
        if (m_freeBlockIndex != INVALID_ID_INDEX) {
            Block& block = m_blocks[m_freeBlockIndex];
            m_freeBlockIndex = setIndexValue(block.index, m_freeBlockIndex);

            blockIndex = block.index;
        } else {
            uint32_t newIndex = (uint32_t)m_blocks.size();
            m_blocks.resize(m_blocks.size() + 1);
            Block& block = m_blocks[newIndex];
            block.index = newIndex;

            blockIndex = newIndex;
        }

        Block& block = m_blocks[blockIndex];
        block.size = std::max(m_blockSize, size);
        if (!isDedicated) {
            // only adjust size if not dedicated.
            // warning this lowers from 64 bit to 32 bit size, which should be fine given
            // such big allocations will trigger the dedicated path
            block.size = block.range.alignedSize((uint32_t)block.size);
        }

        allocBlock(block, blockIndex, block.size);

        block.isDedicated = isDedicated;

        if (!isDedicated) {
            // Dedicated blocks don't allow for subranges, so don't initialize the range allocator
            block.range.init((uint32_t)block.size);
            block.range.subAllocate((uint32_t)size, align, usedOffset, usedAligned, usedSize);
            m_regularBlocks++;
        }
    }

    Handle sub;
    if (!sub.setup(blockIndex, isDedicated ? 0 : usedOffset,
                   isDedicated ? size : uint64_t(usedSize), isDedicated)) {
        return Handle();
    }

    // append used space for stats
    m_usedSize += sub.getSize();

    return sub;
}

void BufferSubAllocator::subFree(Handle sub) {
    if (!sub)
        return;

    Block& block = getBlock(sub.block.blockIndex);
    bool isDedicated = sub.isDedicated();
    if (!isDedicated) {
        block.range.subFree(uint32_t(sub.getOffset()), uint32_t(sub.getSize()));
    }

    m_usedSize -= sub.getSize();

    if (isDedicated || (block.range.isEmpty() && (!m_keepLastBlock || m_regularBlocks > 1))) {
        if (!isDedicated) {
            m_regularBlocks--;
        }
        freeBlock(block);
    }
}

float BufferSubAllocator::getUtilization(VkDeviceSize& allocatedSize,
                                         VkDeviceSize& usedSize) const {
    allocatedSize = m_allocatedSize;
    usedSize = m_usedSize;

    return float(double(usedSize) / double(allocatedSize));
}

bool BufferSubAllocator::fitsInAllocated(VkDeviceSize size, uint32_t alignment) const {
    if (Handle::needsDedicated(size, alignment)) {
        return false;
    }

    for (const auto& block : m_blocks) {
        if (block.buffer && !block.isDedicated) {
            if (block.range.isAvailable((uint32_t)size, (uint32_t)alignment)) {
                return true;
            }
        }
    }

    return false;
}

void BufferSubAllocator::free(bool onlyEmpty) {
    for (uint32_t i = 0; i < (uint32_t)m_blocks.size(); i++) {
        Block& block = m_blocks[i];
        if (block.buffer && (!onlyEmpty || (!block.isDedicated && block.range.isEmpty()))) {
            freeBlock(block);
        }
    }

    if (!onlyEmpty) {
        m_blocks.clear();
        m_freeBlockIndex = INVALID_ID_INDEX;
    }
}

void BufferSubAllocator::freeBlock(Block& block) {
    m_allocatedSize -= block.size;

    vkDestroyBuffer(m_device, block.buffer, nullptr);
    if (block.mapping) {
        block.memory->unmap();
    }

    if (!block.isDedicated) {
        block.range.deinit();
    }
    block.memory = NullMememoryAllocationHandle;
    block.buffer = VK_NULL_HANDLE;
    block.mapping = nullptr;
    block.isDedicated = false;

    // update the block.index with the current head of the free list
    // pop its old value
    m_freeBlockIndex = setIndexValue(block.index, m_freeBlockIndex);
}

void BufferSubAllocator::allocBlock(Block& block, uint32_t, vk::DeviceSize size) {

    vk::SharingMode sharingMode = m_sharingQueueFamilyIndices.size() > 1
                                      ? vk::SharingMode::eConcurrent
                                      : vk::SharingMode::eExclusive;
    vk::BufferCreateInfo buffer_create_info{
        {}, size, m_bufferUsageFlags, sharingMode, m_sharingQueueFamilyIndices,
    };

    vk::Buffer buffer = m_device.createBuffer(buffer_create_info, nullptr);

    vk::BufferMemoryRequirementsInfo2KHR buffer_requirements{buffer};
    vk::MemoryRequirements2KHR memory_requirements =
        m_device.getBufferMemoryRequirements2(buffer_requirements);

    if (m_memoryTypeIndex == uint32_t(~0)) {
        vk::PhysicalDeviceMemoryProperties memoryProperties =
            m_memAllocator->get_context()
                ->pd_container.physical_device_memory_properties.memoryProperties;
        vk::MemoryPropertyFlags memProps = m_memoryPropFlags;

        // Find an available memory type that satisfies the requested properties.
        for (uint32_t memoryTypeIndex = 0; memoryTypeIndex < memoryProperties.memoryTypeCount;
             ++memoryTypeIndex) {
            if ((memory_requirements.memoryRequirements.memoryTypeBits & (1 << memoryTypeIndex)) &&
                (memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags & memProps) ==
                    memProps) {
                m_memoryTypeIndex = memoryTypeIndex;
                break;
            }
        }
    }

    if (m_memoryTypeIndex == uint32_t(~0)) {
        m_device.destroyBuffer(buffer);
        throw std::runtime_error("could not find memoryTypeIndex\n");
    }

    // TODO: Really only sequential?!
    MemoryAllocationHandle memory =
        m_memAllocator->allocate_memory(m_memoryPropFlags, memory_requirements.memoryRequirements,
                                        "suballocator buffer", HOST_ACCESS_SEQUENTIAL_WRITE);
    MemoryAllocation::MemoryInfo memInfo = memory->get_memory_info();

    vk::BindBufferMemoryInfo bindInfos{buffer, memInfo.memory, memInfo.offset};
    m_device.bindBufferMemory2({bindInfos});

    if (m_mapped) {
        block.mapping = memory->map_as<uint8_t>();
    } else {
        block.mapping = nullptr;
    }

    if (!m_mapped || block.mapping) {
        if (m_bufferUsageFlags & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            vk::BufferDeviceAddressInfoEXT info{buffer};
            block.address = m_device.getBufferAddress(info);
        }

        block.memory = memory;
        block.buffer = buffer;
        m_allocatedSize += block.size;
    }
}

} // namespace merian
