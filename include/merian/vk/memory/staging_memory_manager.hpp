#pragma once

#include "merian/vk/memory/buffer_suballocator.hpp"

#include <vulkan/vulkan.hpp>

#include <string>
#include <vector>

namespace merian {

#define NVVK_DEFAULT_STAGING_BLOCKSIZE (vk::DeviceSize(64) * 1024 * 1024)

//////////////////////////////////////////////////////////////////
/**
  \class merian::StagingMemoryManager

  merian::StagingMemoryManager class is a utility that manages host visible
  buffers and their allocations in an opaque fashion to assist
  asynchronous transfers between device and host.
  The memory for this is allocated using the provided
  [merian::MemAllocator](#class-merianmemallocator).

  The collection of the transfer resources is represented by merian::StagingID.

  The necessary buffer space is sub-allocated and recycled by using one
  [merian::BufferSubAllocator](#class-merianbuffersuballocator) per transfer direction (to or from device).

  > **WARNING:**
  > - cannot manage a copy > 4 GB

  Usage:
  - Enqueue transfers into your vk::CommandBuffer and then finalize the copy operations.
  - Associate the copy operations with a vk::Fence or retrieve a SetID
  - The release of the resources allows to safely recycle the buffer space for future transfers.

  > We use fences as a way to garbage collect here, however a more robust solution
  > may be implementing some sort of ticketing/timeline system.
  > If a fence is recycled, then this class may not be aware that the fence represents a different
  > submission, likewise if the fence is deleted elsewhere problems can occur.
  > You may want to use the manual "SetID" system in that case.

  Example :

  \code{.cpp}
  StagingMemoryManager  staging;
  staging.init(memAllocator);


  // Enqueue copy operations of data to target buffer.
  // This internally manages the required staging resources
  staging.cmdToBuffer(cmd, targetBufer, 0, targetSize, targetData);

  // you can also get access to a temporary mapped pointer and fill
  // the staging buffer directly
  vertices = staging.cmdToBufferT<Vertex>(cmd, targetBufer, 0, targetSize);

  // OPTION A:
  // associate all previous copy operations with a fence (or not)
  staging.finalizeResources( fence );
  ..
  // every once in a while call
  staging.releaseResources();
  // this will release all those without fence, or those
  // who had a fence that completed (but never manual SetIDs, see next).

  // OPTION B
  // alternatively manage the resource release yourself.
  // The SetID represents the staging resources
  // since any last finalize.
  sid = staging.finalizeResourceSet();

  ...
  // You need to ensure these transfers and their staging
  // data access completed yourself prior releasing the set.
  //
  // This is particularly useful for managing downloads from
  // device. The "from" functions return a pointer  where the
  // data will be copied to. You want to use this pointer
  // after the device-side transfer completed, and then
  // release its resources once you are done using it.

  staging.releaseResourceSet(sid);

  \endcode
*/

class StagingMemoryManager {
  public:
    static const uint32_t INVALID_ID_INDEX = ~0;

    //////////////////////////////////////////////////////////////////////////
    class SetID {
        friend StagingMemoryManager;

      private:
        uint32_t index = INVALID_ID_INDEX;
    };

    StagingMemoryManager(StagingMemoryManager const&) = delete;
    StagingMemoryManager& operator=(StagingMemoryManager const&) = delete;

    StagingMemoryManager() {
    }
    StagingMemoryManager(MemoryAllocator* memAllocator, vk::DeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE) {
        init(memAllocator, stagingBlockSize);
    }

    virtual ~StagingMemoryManager() {
        deinit();
    }

    void init(MemoryAllocator* memAllocator, vk::DeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE);
    void deinit();

    // if true (default) we free the memory completely when released
    // otherwise we would keep blocks for re-use around, unless freeUnused() is called
    void setFreeUnusedOnRelease(bool state) {
        m_subToDevice.setKeepLastBlockOnFree(!state);
        m_subFromDevice.setKeepLastBlockOnFree(!state);
    }

    // test if there is enough space in current allocations
    bool fitsInAllocated(vk::DeviceSize size, bool toDevice = true) const;

    // if data != nullptr memcpies to mapping and returns nullptr
    // otherwise returns temporary mapping (valid until "complete" functions)
    void* cmdToImage(vk::CommandBuffer cmd, vk::Image image, const vk::Offset3D& offset, const vk::Extent3D& extent,
                     const vk::ImageSubresourceLayers& subresource, vk::DeviceSize size, const void* data,
                     vk::ImageLayout layout = vk::ImageLayout::eTransferDstOptimal);

    template <class T>
    T* cmdToImageT(vk::CommandBuffer cmd, vk::Image image, const vk::Offset3D& offset, const vk::Extent3D& extent,
                   const vk::ImageSubresourceLayers& subresource, vk::DeviceSize size, const void* data,
                   vk::ImageLayout layout = vk::ImageLayout::eTransferDstOptimal) {
        return (T*)cmdToImage(cmd, image, offset, extent, subresource, size, data, layout);
    }

    // pointer can be used after cmd execution but only valid until associated resources haven't been released
    const void* cmdFromImage(vk::CommandBuffer cmd, vk::Image image, const vk::Offset3D& offset, const vk::Extent3D& extent,
                             const vk::ImageSubresourceLayers& subresource, vk::DeviceSize size,
                             vk::ImageLayout layout = vk::ImageLayout::eTransferSrcOptimal);

    template <class T>
    const T* cmdFromImageT(vk::CommandBuffer cmd, vk::Image image, const vk::Offset3D& offset, const vk::Extent3D& extent,
                           const vk::ImageSubresourceLayers& subresource, vk::DeviceSize size,
                           vk::ImageLayout layout = vk::ImageLayout::eTransferSrcOptimal) {
        return (const T*)cmdFromImage(cmd, image, offset, extent, subresource, size, layout);
    }

    // if data != nullptr memcpies to mapping and returns nullptr
    // otherwise returns temporary mapping (valid until appropriate release)
    void* cmdToBuffer(vk::CommandBuffer cmd, vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size, const void* data);

    template <class T> T* cmdToBufferT(vk::CommandBuffer cmd, vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size) {
        return (T*)cmdToBuffer(cmd, buffer, offset, size, nullptr);
    }

    // pointer can be used after cmd execution but only valid until associated resources haven't been released
    const void* cmdFromBuffer(vk::CommandBuffer cmd, vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size);

    template <class T>
    const T* cmdFromBufferT(vk::CommandBuffer cmd, vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size) {
        return (const T*)cmdFromBuffer(cmd, buffer, offset, size);
    }

    // closes the batch of staging resources since last finalize call
    // and associates it with a fence for later release.
    void finalizeResources(vk::Fence fence = VK_NULL_HANDLE);

    // releases the staging resources whose fences have completed
    // and those who had no fence at all, skips resourceSets.
    void releaseResources();

    // closes the batch of staging resources since last finalize call
    // and returns a resource set handle that can be used to release them
    SetID finalizeResourceSet();

    // releases the staging resources from this particular
    // resource set.
    void releaseResourceSet(SetID setid) {
        releaseResources(setid.index);
    }

    // frees staging memory no longer in use
    void freeUnused() {
        free(true);
    }

    float getUtilization(vk::DeviceSize& allocatedSize, vk::DeviceSize& usedSize) const;

  protected:
    // The implementation uses two major arrays:
    // - Block stores vk::Buffers that we sub-allocate the staging space from
    // - StagingSet stores all such sub-allocations that were used
    //   in one batch of operations. Each batch is closed with
    //   finalizeResources, and typically associated with a fence.
    //   As such the resources are given by for recycling if the fence completed.

    // To recycle StagingSet structures within the arrays
    // we use a linked list of array indices. The "index" element
    // in the struct refers to the next free list item, or itself
    // when in use.

    struct Entry {
        BufferSubAllocator::Handle handle;
        bool toDevice;
    };

    struct StagingSet {
        uint32_t index = INVALID_ID_INDEX;
        vk::Fence fence = VK_NULL_HANDLE;
        bool manualSet = false;
        std::vector<Entry> entries;
    };

    vk::Device m_device = VK_NULL_HANDLE;

    BufferSubAllocator m_subToDevice;
    BufferSubAllocator m_subFromDevice;

    std::vector<StagingSet> m_sets;

    // active staging Index, must be valid at all items
    uint32_t m_stagingIndex;
    // linked-list to next free staging set
    uint32_t m_freeStagingIndex;

    uint32_t setIndexValue(uint32_t& index, uint32_t newValue) {
        uint32_t oldValue = index;
        index = newValue;
        return oldValue;
    }

    void free(bool unusedOnly);

    uint32_t newStagingIndex();

    void* getStagingSpace(vk::DeviceSize size, vk::Buffer& buffer, vk::DeviceSize& offset, bool toDevice);

    void releaseResources(uint32_t stagingID);
};

} // namespace merian
