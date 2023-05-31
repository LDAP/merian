#include "merian/vk/memory/memory_allocator_vma.hpp"
#include "merian/utils/debug.hpp"
#include "merian/vk/utils/check_result.hpp"

namespace merian {

//--------------------------------------------------------------------------------------------------
// Converter utility from Vulkan memory property to VMA
//
VmaMemoryUsage vkToVmaMemoryUsage(vk::MemoryPropertyFlags flags)

{
    if ((flags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal)
        return VMA_MEMORY_USAGE_GPU_ONLY;
    else if ((flags & vk::MemoryPropertyFlagBits::eHostCoherent) == vk::MemoryPropertyFlagBits::eHostCoherent)
        return VMA_MEMORY_USAGE_CPU_ONLY;
    else if ((flags & vk::MemoryPropertyFlagBits::eHostVisible) == vk::MemoryPropertyFlagBits::eHostVisible)
        return VMA_MEMORY_USAGE_CPU_TO_GPU;
    return VMA_MEMORY_USAGE_UNKNOWN;
}

VMAMemoryHandle* castVMAMemoryHandle(MemHandle memHandle) {
    if (!memHandle)
        return nullptr;

#ifdef DEBUG
    auto vmaMemHandle = static_cast<VMAMemoryHandle*>(memHandle);
#else
    auto vmaMemHandle = dynamic_cast<VMAMemoryHandle*>(memHandle);
    assert(vmaMemHandle);
#endif

    return vmaMemHandle;
}

VMAMemoryAllocator::VMAMemoryAllocator(vk::Device device, vk::PhysicalDevice physicalDevice, VmaAllocator vma) {
    init(device, physicalDevice, vma);
}

VMAMemoryAllocator::~VMAMemoryAllocator() {
    deinit();
}

bool VMAMemoryAllocator::init(vk::Device device, vk::PhysicalDevice physicalDevice, VmaAllocator vma) {
    m_device = device;
    m_physicalDevice = physicalDevice;
    m_vma = vma;
    return true;
}

void VMAMemoryAllocator::deinit() {
    m_vma = 0;
}

MemHandle VMAMemoryAllocator::allocMemory(const MemAllocateInfo& allocInfo, vk::Result* pResult) {
    VmaAllocationCreateInfo vmaAllocInfo = {};
    vmaAllocInfo.usage = vkToVmaMemoryUsage(allocInfo.getMemoryProperties());
    if (allocInfo.getDedicatedBuffer() || allocInfo.getDedicatedImage()) {
        vmaAllocInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }
    vmaAllocInfo.priority = allocInfo.getPriority();

    // Not supported by VMA
    assert(!allocInfo.getExportable());
    assert(!allocInfo.getDeviceMask());

    VmaAllocationInfo allocationDetail;
    VmaAllocation allocation = nullptr;

    VkMemoryRequirements mem_reqs = allocInfo.getMemoryRequirements();
    VkResult result = vmaAllocateMemory(m_vma, &mem_reqs, &vmaAllocInfo, &allocation, &allocationDetail);

#ifdef DEBUG
    // !! VMA leaks finder!!
    // Call findLeak with the value showing in the leak report.
    // Add : #define VMA_DEBUG_LOG(format, ...) do { printf(format, __VA_ARGS__); printf("\n"); } while(false)
    //  - in the app where VMA_IMPLEMENTATION is defined, to have a leak report
    static uint64_t counter{0};
    if (counter == m_leakID) {
        debugbreak();
    }
    if (result == VK_SUCCESS) {
        std::string allocID = std::to_string(counter++);
        vmaSetAllocationName(m_vma, allocation, allocID.c_str());
    }
#endif // DEBUG

    check_result(result, "could not allocate memory");

    if (pResult) {
        *pResult = vk::Result(result);
    }
    return new VMAMemoryHandle(allocation);
}

void VMAMemoryAllocator::freeMemory(MemHandle memHandle) {
    if (!memHandle)
        return;

    auto vmaHandle = castVMAMemoryHandle(memHandle);
    vmaFreeMemory(m_vma, vmaHandle->getAllocation());
}

MemoryAllocator::MemInfo VMAMemoryAllocator::getMemoryInfo(MemHandle memHandle) const {
    auto vmaHandle = castVMAMemoryHandle(memHandle);

    VmaAllocationInfo allocInfo;
    vmaGetAllocationInfo(m_vma, vmaHandle->getAllocation(), &allocInfo);

    MemInfo memInfo;
    memInfo.memory = allocInfo.deviceMemory;
    memInfo.offset = allocInfo.offset;
    memInfo.size = allocInfo.size;

    return memInfo;
}

void* VMAMemoryAllocator::map(MemHandle memHandle, vk::DeviceSize offset, vk::DeviceSize size, vk::Result* pResult) {
    auto vmaHandle = castVMAMemoryHandle(memHandle);

    void* ptr;
    VkResult result = vmaMapMemory(m_vma, vmaHandle->getAllocation(), &ptr);
    check_result(result, "mapping memory failed");

    if (pResult) {
        *pResult = vk::Result(result);
    }

    return ptr;
}

void VMAMemoryAllocator::unmap(MemHandle memHandle) {
    auto vmaHandle = castVMAMemoryHandle(memHandle);

    vmaUnmapMemory(m_vma, vmaHandle->getAllocation());
}

vk::Device VMAMemoryAllocator::getDevice() const {
    return m_device;
}

vk::PhysicalDevice VMAMemoryAllocator::getPhysicalDevice() const {
    return m_physicalDevice;
}

} // namespace merian
