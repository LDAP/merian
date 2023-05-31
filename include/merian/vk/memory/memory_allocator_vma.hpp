#pragma once

#include "merian/vk/memory/memory_allocator.hpp"

#include <cstdio>
#include <vk_mem_alloc.h>

namespace merian {

class VMAMemoryHandle : public MemHandleBase {
  public:
    VMAMemoryHandle() = default;
    VMAMemoryHandle(const VMAMemoryHandle&) = default;
    VMAMemoryHandle(VMAMemoryHandle&&) = default;

    VmaAllocation getAllocation() const {
        return m_allocation;
    }

  private:
    friend class VMAMemoryAllocator;
    VMAMemoryHandle(VmaAllocation allocation) : m_allocation(allocation) {}

    VmaAllocation m_allocation;
};

class VMAMemoryAllocator : public MemoryAllocator {
  public:
    VMAMemoryAllocator() = default;
    
    explicit VMAMemoryAllocator(vk::Device device, vk::PhysicalDevice physicalDevice, VmaAllocator vma);
    virtual ~VMAMemoryAllocator();

    bool init(vk::Device device, vk::PhysicalDevice physicalDevice, VmaAllocator vma);
    void deinit();

    MemHandle allocMemory(const MemAllocateInfo& allocInfo, vk::Result* pResult = nullptr) override;
    void freeMemory(MemHandle memHandle) override;
    MemInfo getMemoryInfo(MemHandle memHandle) const override;
    void* map(MemHandle memHandle, vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE,
              vk::Result* pResult = nullptr) override;
    void unmap(MemHandle memHandle) override;

    vk::Device getDevice() const override;
    vk::PhysicalDevice getPhysicalDevice() const override;

    void findLeak(uint64_t leakID) {
        m_leakID = leakID;
    }

  private:
    VmaAllocator m_vma{0};
    vk::Device m_device{nullptr};
    vk::PhysicalDevice m_physicalDevice{nullptr};
    uint64_t m_leakID{~0U};
};

} // namespace merian
