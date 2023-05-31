#pragma once

#include <vulkan/vulkan.hpp>

#include <string>

namespace merian {

/*
MemHandle represents a memory allocation or sub-allocation from the
generic merian::MemAllocator interface. Ideally use `merian::NullMemHandle` for
setting to 'NULL'.

\class merian::MemAllocateInfo

merian::MemAllocateInfo is collecting almost all parameters a Vulkan allocation could potentially need.
This keeps MemAllocator's interface simple and extensible.
*/

class MemHandleBase;
typedef MemHandleBase* MemHandle;
static const MemHandle NullMemHandle = nullptr;

class MemAllocateInfo {
  public:
    MemAllocateInfo(
        const vk::MemoryRequirements& memReqs, // determine size, alignment and memory type
        vk::MemoryPropertyFlags memProps =
            vk::MemoryPropertyFlagBits::eDeviceLocal, // determine device_local, host_visible, host coherent etc...
        bool isTilingOptimal =
            false // determine if the alocation is going to be used for an VK_IMAGE_TILING_OPTIMAL image
    );

    // Convenience constructures that infer the allocation information from the buffer object directly
    MemAllocateInfo(vk::Device device, vk::Buffer buffer,
                    vk::MemoryPropertyFlags memProps = vk::MemoryPropertyFlagBits::eDeviceLocal);
    // Convenience constructures that infer the allocation information from the image object directly.
    // If the driver _prefers_ a dedicated allocation for this particular image and allowDedicatedAllocation is true, a
    // dedicated allocation will be requested. If the driver _requires_ a dedicated allocation, a dedicated allocation
    // will be requested regardless of 'allowDedicatedAllocation'.
    MemAllocateInfo(vk::Device device, vk::Image image,
                    vk::MemoryPropertyFlags memProps = vk::MemoryPropertyFlagBits::eDeviceLocal,
                    bool allowDedicatedAllocation = true);

    // Determines which heap to allocate from
    MemAllocateInfo& setMemoryProperties(vk::MemoryPropertyFlags flags);
    // Determines size and alignment
    MemAllocateInfo& setMemoryRequirements(vk::MemoryRequirements requirements);
    // TilingOptimal should be set for images. The allocator may choose to separate linear and tiling allocations
    MemAllocateInfo& setTilingOptimal(bool isTilingOptimal);
    // The allocation will be dedicated for the given image
    MemAllocateInfo& setDedicatedImage(vk::Image image);
    // The allocation will be dedicated for the given buffer
    MemAllocateInfo& setDedicatedBuffer(vk::Buffer buffer);
    // Set additional allocation flags
    MemAllocateInfo& setAllocationFlags(vk::MemoryAllocateFlags flags);
    // Set the device mask for the allocation, redirect allocations to specific device(s) in the device group
    MemAllocateInfo& setDeviceMask(uint32_t mask);
    // Set a name for the allocation (only useful for dedicated allocations or allocators)
    MemAllocateInfo& setDebugName(const std::string& name);
    // Make the allocation exportable
    MemAllocateInfo& setExportable(bool exportable);
    // Prioritize the allocation (values 0.0 - 1.0); this may guide eviction strategies
    MemAllocateInfo& setPriority(const float priority = 0.5f);

    vk::Image getDedicatedImage() const {
        return m_dedicatedImage;
    }
    vk::Buffer getDedicatedBuffer() const {
        return m_dedicatedBuffer;
    }
    vk::MemoryAllocateFlags getAllocationFlags() const {
        return m_allocateFlags;
    }
    uint32_t getDeviceMask() const {
        return m_deviceMask;
    }
    bool getTilingOptimal() const {
        return m_isTilingOptimal;
    }
    const vk::MemoryRequirements& getMemoryRequirements() const {
        return m_memReqs;
    }
    const vk::MemoryPropertyFlags& getMemoryProperties() const {
        return m_memProps;
    }
    std::string getDebugName() const {
        return m_debugName;
    }
    bool getExportable() const {
        return m_isExportable;
    }
    float getPriority() const {
        return m_priority;
    }

  private:
    vk::Buffer m_dedicatedBuffer{VK_NULL_HANDLE};
    vk::Image m_dedicatedImage{VK_NULL_HANDLE};
    vk::MemoryAllocateFlags m_allocateFlags{0};
    uint32_t m_deviceMask{0};
    vk::MemoryRequirements m_memReqs{0, 0, 0};
    vk::MemoryPropertyFlags m_memProps{0};
    float m_priority{0.5f};

    std::string m_debugName;

    bool m_isTilingOptimal{false};
    bool m_isExportable{false};
};

// BakedAllocateInfo is a group of allocation relevant Vulkan allocation structures,
// which will be filled out and linked via pNext-> to be used directly via vkAllocateMemory.
struct BakedAllocateInfo {
    BakedAllocateInfo() = default;

    // In lieu of proper copy operators, need to delete them as we store
    // addresses to members in other members. Copying such object would make them point to
    // wrong or out-of-scope addresses
    BakedAllocateInfo(BakedAllocateInfo&& other) = delete;
    BakedAllocateInfo operator=(BakedAllocateInfo&& other) = delete;
    BakedAllocateInfo(const BakedAllocateInfo&) = delete;
    BakedAllocateInfo operator=(const BakedAllocateInfo) = delete;

    vk::MemoryAllocateInfo memAllocInfo;
    vk::MemoryAllocateFlagsInfo flagsInfo;
    vk::MemoryDedicatedAllocateInfo dedicatedInfo;
    vk::ExportMemoryAllocateInfo exportInfo;
};

bool fillBakedAllocateInfo(const vk::PhysicalDeviceMemoryProperties& physMemProps, const MemAllocateInfo& info,
                           BakedAllocateInfo& baked);
uint32_t getMemoryType(const vk::PhysicalDeviceMemoryProperties& memoryProperties, uint32_t typeBits,
                       const vk::MemoryPropertyFlags& properties);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
  \class merian::MemAllocator

 merian::MemAllocator is a Vulkan memory allocator interface extensively used by ResourceAllocator.
 It provides means to allocate, free, map and unmap pieces of Vulkan device memory.
 Concrete implementations derive from merian::MemoryAllocator.
 They can implement the allocator dunctionality themselves or act as an adapter to another
 memory allocator implementation.

 A merian::MemAllocator hands out opaque 'MemHandles'. The implementation of the MemAllocator interface
 may chose any type of payload to store in a MemHandle. A MemHandle's relevant information can be
 retrieved via getMemoryInfo().
*/
class MemoryAllocator {
  public:
    struct MemInfo {
        vk::DeviceMemory memory;
        vk::DeviceSize offset;
        vk::DeviceSize size;
    };

    // Allocate a piece of memory according to the requirements of allocInfo.
    // may return NullMemHandle on error (provide pResult for details)
    virtual MemHandle allocMemory(const MemAllocateInfo& allocInfo, vk::Result* pResult = nullptr) = 0;

    // Free the memory backing 'memHandle'.
    // memHandle may be nullptr;
    virtual void freeMemory(MemHandle memHandle) = 0;

    // Retrieve detailed information about 'memHandle'
    virtual MemInfo getMemoryInfo(MemHandle memHandle) const = 0;

    // Maps device memory to system memory.
    // If 'memHandle' already refers to a suballocation 'offset' will be applied on top of the
    // suballocation's offset inside the device memory.
    // may return nullptr on error (provide pResult for details)
    virtual void* map(MemHandle memHandle, vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE,
                      vk::Result* pResult = nullptr) = 0;

    // Unmap memHandle
    virtual void unmap(MemHandle memHandle) = 0;

    // Convenience function to allow mapping straight to a typed pointer.
    template <class T> T* mapT(MemHandle memHandle, vk::Result* pResult = nullptr) {
        return (T*)map(memHandle, 0, VK_WHOLE_SIZE, pResult);
    }

    virtual vk::Device getDevice() const = 0;
    virtual vk::PhysicalDevice getPhysicalDevice() const = 0;

    // Make sure the dtor is virtual
    virtual ~MemoryAllocator() = default;
};

// Base class for memory handles
// Individual allocators will derive from it and fill the handles with their own data.
class MemHandleBase {
  public:
    virtual ~MemHandleBase() = default; // force the class to become virtual
};

} // namespace merian
