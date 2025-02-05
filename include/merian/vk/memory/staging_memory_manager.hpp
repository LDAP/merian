#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/memory/memory_allocator.hpp"

#include "merian/vk/utils/check_result.hpp"
#include "vk_mem_alloc.h"

#include <map>

namespace merian {

class StagingMemoryManager : public std::enable_shared_from_this<StagingMemoryManager> {
  public:
    class Allocation {

      public:
        Allocation(const BufferHandle& buffer) : buffer(buffer) {
            VmaVirtualBlockCreateInfo block_create_info = {};
            block_create_info.size = buffer->get_size();
            check_result(vmaCreateVirtualBlock(&block_create_info, &block),
                         "could not create virtual allocation.");
        }

        ~Allocation() {
            vmaClearVirtualBlock(block);
            vmaDestroyVirtualBlock(block);
        }

      private:
        BufferHandle buffer;
        VmaVirtualBlock block;
    };

    class StagingSet {
        friend StagingMemoryManager;

      private:
        // vk::DeviceSize: estimated free size, random counter to prevent map collisions
        std::map<std::pair<vk::DeviceSize, uint64_t>, Allocation> allocations_to_gpu;
        std::map<std::pair<vk::DeviceSize, uint64_t>, Allocation> allocations_from_gpu;
    };

    StagingMemoryManager(StagingMemoryManager const&) = delete;
    StagingMemoryManager& operator=(StagingMemoryManager const&) = delete;
    StagingMemoryManager() = delete;
    StagingMemoryManager(const MemoryAllocatorHandle& memory_allocator,
                         const vk::DeviceSize stagingBlockSize = (vk::DeviceSize(128) * 1024 *
                                                                  1024));

    ~StagingMemoryManager();

    // -------------------------------------------------------------------------
    
    // Start a new staging set. Call this at the 
    void new_staging_set();

    // -------------------------------------------------------------------------

    // if data != nullptr memcpies to mapping and returns nullptr
    // otherwise returns temporary mapping
    void* cmdToImage(const CommandBufferHandle& cmd,
                     vk::Image image,
                     const vk::Offset3D& offset,
                     const vk::Extent3D& extent,
                     const vk::ImageSubresourceLayers& subresource,
                     vk::DeviceSize size,
                     const void* data,
                     vk::ImageLayout layout = vk::ImageLayout::eTransferDstOptimal);

    template <class T>
    T* cmdToImageT(const CommandBufferHandle& cmd,
                   vk::Image image,
                   const vk::Offset3D& offset,
                   const vk::Extent3D& extent,
                   const vk::ImageSubresourceLayers& subresource,
                   vk::DeviceSize size,
                   const void* data,
                   vk::ImageLayout layout = vk::ImageLayout::eTransferDstOptimal) {
        return (T*)cmdToImage(cmd, image, offset, extent, subresource, size, data, layout);
    }

    // pointer can be used after cmd execution but only valid until associated resources haven't
    // been released
    const void* cmdFromImage(const CommandBufferHandle& cmd,
                             vk::Image image,
                             const vk::Offset3D& offset,
                             const vk::Extent3D& extent,
                             const vk::ImageSubresourceLayers& subresource,
                             vk::DeviceSize size,
                             vk::ImageLayout layout = vk::ImageLayout::eTransferSrcOptimal);

    template <class T>
    const T* cmdFromImageT(const CommandBufferHandle& cmd,
                           vk::Image image,
                           const vk::Offset3D& offset,
                           const vk::Extent3D& extent,
                           const vk::ImageSubresourceLayers& subresource,
                           vk::DeviceSize size,
                           vk::ImageLayout layout = vk::ImageLayout::eTransferSrcOptimal) {
        return (const T*)cmdFromImage(cmd, image, offset, extent, subresource, size, layout);
    }

    // if data != nullptr memcpies to mapping and returns nullptr
    // otherwise returns temporary mapping (valid until appropriate release)
    void* cmdToBuffer(const CommandBufferHandle& cmd,
                      vk::Buffer buffer,
                      vk::DeviceSize offset,
                      vk::DeviceSize size,
                      const void* data);

    template <class T>
    T* cmdToBufferT(const CommandBufferHandle& cmd,
                    vk::Buffer buffer,
                    vk::DeviceSize offset,
                    vk::DeviceSize size) {
        return (T*)cmdToBuffer(cmd, buffer, offset, size, nullptr);
    }

    // pointer can be used after cmd execution but only valid until associated resources haven't
    // been released
    const void* cmdFromBuffer(const CommandBufferHandle& cmd,
                              vk::Buffer buffer,
                              vk::DeviceSize offset,
                              vk::DeviceSize size);

    template <class T>
    const T* cmdFromBufferT(const CommandBufferHandle& cmd,
                            vk::Buffer buffer,
                            vk::DeviceSize offset,
                            vk::DeviceSize size) {
        return (const T*)cmdFromBuffer(cmd, buffer, offset, size);
    }

    // -------------------------------------------------------------------------

  private:
    const ContextHandle context;
    const MemoryAllocatorHandle allocator;

    StagingSet current_staging_set;
};

using StagingMemoryManagerHandle = std::shared_ptr<StagingMemoryManager>;

} // namespace merian
