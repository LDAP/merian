#include <vk/memory/staging_memory_manager.hpp>

namespace merian {

void StagingMemoryManager::init(MemoryAllocator* memAllocator, vk::DeviceSize stagingBlockSize /*= 64 * 1024 * 1024*/) {
    assert(!m_device);
    m_device = memAllocator->getDevice();

    m_subToDevice.init(memAllocator, stagingBlockSize, vk::BufferUsageFlagBits::eTransferSrc,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, true);
    m_subFromDevice.init(memAllocator, stagingBlockSize, vk::BufferUsageFlagBits::eTransferDst,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent |
                             vk::MemoryPropertyFlagBits::eHostCached,
                         true);

    m_freeStagingIndex = INVALID_ID_INDEX;
    m_stagingIndex = newStagingIndex();

    setFreeUnusedOnRelease(true);
}

void StagingMemoryManager::deinit() {
    if (!m_device)
        return;

    free(false);

    m_subFromDevice.deinit();
    m_subToDevice.deinit();

    m_sets.clear();
    m_device = VK_NULL_HANDLE;
}

bool StagingMemoryManager::fitsInAllocated(vk::DeviceSize size, bool toDevice /*= true*/) const {
    return toDevice ? m_subToDevice.fitsInAllocated(size) : m_subFromDevice.fitsInAllocated(size);
}

void* StagingMemoryManager::cmdToImage(vk::CommandBuffer cmd, vk::Image image, const vk::Offset3D& offset,
                                       const vk::Extent3D& extent, const vk::ImageSubresourceLayers& subresource,
                                       vk::DeviceSize size, const void* data, vk::ImageLayout layout) {
    if (!image)
        return nullptr;

    vk::Buffer srcBuffer;
    vk::DeviceSize srcOffset;

    void* mapping = getStagingSpace(size, srcBuffer, srcOffset, true);

    assert(mapping);

    if (data) {
        memcpy(mapping, data, size);
    }

    vk::BufferImageCopy cpy{srcOffset, 0, 0, subresource, offset, extent};
    cmd.copyBufferToImage(srcBuffer, image, layout, {cpy});

    return data ? nullptr : mapping;
}

void* StagingMemoryManager::cmdToBuffer(vk::CommandBuffer cmd, vk::Buffer buffer, vk::DeviceSize offset,
                                        vk::DeviceSize size, const void* data) {
    if (!size || !buffer) {
        return nullptr;
    }

    vk::Buffer srcBuffer;
    vk::DeviceSize srcOffset;

    void* mapping = getStagingSpace(size, srcBuffer, srcOffset, true);

    assert(mapping);

    if (data) {
        memcpy(mapping, data, size);
    }

    vk::BufferCopy cpy{srcOffset, offset, size};
    cmd.copyBuffer(srcBuffer, buffer, {cpy});

    return data ? nullptr : (void*)mapping;
}

const void* StagingMemoryManager::cmdFromBuffer(vk::CommandBuffer cmd, vk::Buffer buffer, vk::DeviceSize offset,
                                                vk::DeviceSize size) {
    vk::Buffer dstBuffer;
    vk::DeviceSize dstOffset;
    void* mapping = getStagingSpace(size, dstBuffer, dstOffset, false);

    vk::BufferCopy cpy{offset, dstOffset, size};
    cmd.copyBuffer(buffer, dstBuffer, {cpy});

    return mapping;
}

const void* StagingMemoryManager::cmdFromImage(vk::CommandBuffer cmd, vk::Image image, const vk::Offset3D& offset,
                                               const vk::Extent3D& extent,
                                               const vk::ImageSubresourceLayers& subresource, vk::DeviceSize size,
                                               vk::ImageLayout layout) {
    vk::Buffer dstBuffer;
    vk::DeviceSize dstOffset;
    void* mapping = getStagingSpace(size, dstBuffer, dstOffset, false);

    vk::BufferImageCopy cpy{dstOffset, 0, 0, subresource, offset, extent};
    cmd.copyImageToBuffer(image, layout, dstBuffer, {cpy});

    return mapping;
}

void StagingMemoryManager::finalizeResources(vk::Fence fence) {
    if (m_sets[m_stagingIndex].entries.empty())
        return;

    m_sets[m_stagingIndex].fence = fence;
    m_sets[m_stagingIndex].manualSet = false;
    m_stagingIndex = newStagingIndex();
}

StagingMemoryManager::SetID StagingMemoryManager::finalizeResourceSet() {
    SetID setID;

    if (m_sets[m_stagingIndex].entries.empty())
        return setID;

    setID.index = m_stagingIndex;

    m_sets[m_stagingIndex].fence = nullptr;
    m_sets[m_stagingIndex].manualSet = true;
    m_stagingIndex = newStagingIndex();

    return setID;
}

void* StagingMemoryManager::getStagingSpace(vk::DeviceSize size, vk::Buffer& buffer, vk::DeviceSize& offset,
                                            bool toDevice) {
    assert(m_sets[m_stagingIndex].index == m_stagingIndex && "illegal index, did you forget finalizeResources");

    BufferSubAllocator::Handle handle = toDevice ? m_subToDevice.subAllocate(size) : m_subFromDevice.subAllocate(size);
    assert(handle);

    BufferSubAllocator::Binding info =
        toDevice ? m_subToDevice.getSubBinding(handle) : m_subFromDevice.getSubBinding(handle);
    buffer = info.buffer;
    offset = info.offset;

    // append used space to current staging set list
    m_sets[m_stagingIndex].entries.push_back({handle, toDevice});

    return toDevice ? m_subToDevice.getSubMapping(handle) : m_subFromDevice.getSubMapping(handle);
}

void StagingMemoryManager::releaseResources(uint32_t stagingID) {
    if (stagingID == INVALID_ID_INDEX)
        return;

    StagingSet& set = m_sets[stagingID];
    assert(set.index == stagingID);

    // free used allocation ranges
    for (auto& itentry : set.entries) {
        if (itentry.toDevice) {
            m_subToDevice.subFree(itentry.handle);
        } else {
            m_subFromDevice.subFree(itentry.handle);
        }
    }
    set.entries.clear();

    // update the set.index with the current head of the free list
    // pop its old value
    m_freeStagingIndex = setIndexValue(set.index, m_freeStagingIndex);
}

void StagingMemoryManager::releaseResources() {
    for (auto& itset : m_sets) {
        if (!itset.entries.empty() && !itset.manualSet &&
            (!itset.fence || vkGetFenceStatus(m_device, itset.fence) == VK_SUCCESS)) {
            releaseResources(itset.index);
            itset.fence = VK_NULL_HANDLE;
            itset.manualSet = false;
        }
    }
    // special case for ease of use if there is only one
    if (m_stagingIndex == 0 && m_freeStagingIndex == 0) {
        m_freeStagingIndex = setIndexValue(m_sets[0].index, 0);
    }
}

float StagingMemoryManager::getUtilization(vk::DeviceSize& allocatedSize, vk::DeviceSize& usedSize) const {
    vk::DeviceSize aSize = 0;
    vk::DeviceSize uSize = 0;
    m_subFromDevice.getUtilization(aSize, uSize);

    allocatedSize = aSize;
    usedSize = uSize;
    m_subToDevice.getUtilization(aSize, uSize);
    allocatedSize += aSize;
    usedSize += uSize;

    return float(double(usedSize) / double(allocatedSize));
}

void StagingMemoryManager::free(bool unusedOnly) {
    m_subToDevice.free(unusedOnly);
    m_subFromDevice.free(unusedOnly);
}

uint32_t StagingMemoryManager::newStagingIndex() {
    // find free slot
    if (m_freeStagingIndex != INVALID_ID_INDEX) {
        uint32_t newIndex = m_freeStagingIndex;
        // this updates the free link-list
        m_freeStagingIndex = setIndexValue(m_sets[newIndex].index, newIndex);
        assert(m_sets[newIndex].index == newIndex);
        return m_sets[newIndex].index;
    }

    // otherwise push to end
    uint32_t newIndex = (uint32_t)m_sets.size();

    StagingSet info;
    info.index = newIndex;
    m_sets.push_back(info);

    assert(m_sets[newIndex].index == newIndex);
    return newIndex;
}

} // namespace merian
