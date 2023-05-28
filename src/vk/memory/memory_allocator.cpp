#include "vk/memory/memory_allocator.hpp"

#include <cassert>

namespace merian {

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MemAllocateInfo::MemAllocateInfo(const vk::MemoryRequirements& memReqs, vk::MemoryPropertyFlags memProps,
                                 bool isTilingOptimal)
    : m_memReqs(memReqs), m_memProps(memProps), m_isTilingOptimal(isTilingOptimal) {}

MemAllocateInfo::MemAllocateInfo(vk::Device device, vk::Buffer buffer, vk::MemoryPropertyFlags memProps) {
    vk::BufferMemoryRequirementsInfo2 bufferReqs{buffer};
    vk::MemoryDedicatedRequirements dedicatedRegs;
    vk::MemoryRequirements2 memReqs{{}, &dedicatedRegs};

    device.getBufferMemoryRequirements2(&bufferReqs, &memReqs);

    m_memReqs = memReqs.memoryRequirements;
    m_memProps = memProps;

    if (dedicatedRegs.requiresDedicatedAllocation) {
        setDedicatedBuffer(buffer);
    }

    setTilingOptimal(false);
}

MemAllocateInfo::MemAllocateInfo(vk::Device device, vk::Image image, vk::MemoryPropertyFlags memProps,
                                 bool allowDedicatedAllocation) {
    vk::ImageMemoryRequirementsInfo2 imageReqs{image};
    vk::MemoryDedicatedRequirements dedicatedRegs;
    vk::MemoryRequirements2 memReqs{{}, &dedicatedRegs};

    device.getImageMemoryRequirements2(&imageReqs, &memReqs);

    m_memReqs = memReqs.memoryRequirements;
    m_memProps = memProps;

    if (dedicatedRegs.requiresDedicatedAllocation ||
        (dedicatedRegs.prefersDedicatedAllocation && allowDedicatedAllocation)) {
        setDedicatedImage(image);
    }

    setTilingOptimal(true);
}

MemAllocateInfo& MemAllocateInfo::setDedicatedImage(vk::Image image) {
    assert(!m_dedicatedBuffer);
    m_dedicatedImage = image;

    return *this;
}
MemAllocateInfo& MemAllocateInfo::setDedicatedBuffer(vk::Buffer buffer) {
    assert(!m_dedicatedImage);
    m_dedicatedBuffer = buffer;

    return *this;
}
MemAllocateInfo& MemAllocateInfo::setAllocationFlags(vk::MemoryAllocateFlags flags) {
    m_allocateFlags |= flags;
    return *this;
}

MemAllocateInfo& MemAllocateInfo::setDeviceMask(uint32_t mask) {
    m_deviceMask = mask;
    return *this;
}

MemAllocateInfo& MemAllocateInfo::setDebugName(const std::string& name) {
    m_debugName = name;
    return *this;
}

MemAllocateInfo& MemAllocateInfo::setExportable(bool exportable) {
    m_isExportable = exportable;
    return *this;
}

// Determines which heap to allocate from
MemAllocateInfo& MemAllocateInfo::setMemoryProperties(vk::MemoryPropertyFlags flags) {
    m_memProps = flags;
    return *this;
}
// Determines size and alignment
MemAllocateInfo& MemAllocateInfo::setMemoryRequirements(vk::MemoryRequirements requirements) {
    m_memReqs = requirements;
    return *this;
}

MemAllocateInfo& MemAllocateInfo::setTilingOptimal(bool isTilingOptimal) {
    m_isTilingOptimal = isTilingOptimal;
    return *this;
}

MemAllocateInfo& MemAllocateInfo::setPriority(const float priority /*= 0.5f*/) {
    m_priority = priority;
    return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t getMemoryType(const vk::PhysicalDeviceMemoryProperties& memoryProperties, uint32_t typeBits,
                       const vk::MemoryPropertyFlags& properties) {
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if (((typeBits & (1 << i)) > 0) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    assert(0);
    return ~0u;
}

bool fillBakedAllocateInfo(const vk::PhysicalDeviceMemoryProperties& physMemProps, const MemAllocateInfo& info,
                           BakedAllocateInfo& baked) {
    baked.memAllocInfo.allocationSize = info.getMemoryRequirements().size;
    baked.memAllocInfo.memoryTypeIndex =
        getMemoryType(physMemProps, info.getMemoryRequirements().memoryTypeBits, info.getMemoryProperties());

    // Put it last in the chain, so we can directly pass it into the DeviceMemoryAllocator::alloc function
    if (info.getDedicatedBuffer() || info.getDedicatedImage()) {
        baked.dedicatedInfo.pNext = baked.memAllocInfo.pNext;
        baked.memAllocInfo.pNext = &baked.dedicatedInfo;

        baked.dedicatedInfo.buffer = info.getDedicatedBuffer();
        baked.dedicatedInfo.image = info.getDedicatedImage();
    }

    if (info.getExportable()) {
        baked.exportInfo.pNext = baked.memAllocInfo.pNext;
        baked.memAllocInfo.pNext = &baked.exportInfo;
        baked.exportInfo.handleTypes =  vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
    }

    if (info.getDeviceMask() || info.getAllocationFlags()) {
        baked.flagsInfo.pNext = baked.memAllocInfo.pNext;
        baked.memAllocInfo.pNext = &baked.flagsInfo;

        baked.flagsInfo.flags = info.getAllocationFlags();
        baked.flagsInfo.deviceMask = info.getDeviceMask();

        if (baked.flagsInfo.deviceMask) {
            baked.flagsInfo.flags |= vk::MemoryAllocateFlagBits::eDeviceMask;
        }
    }

    return true;
}

} // namespace merian
