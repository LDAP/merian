#pragma once

#include "merian/utils/range_allocator.hpp"
#include "merian/vk/memory/memory_allocator.hpp"

#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace merian {

//////////////////////////////////////////////////////////////////
/**
  \class merian::BufferSubAllocator

  merian::BufferSubAllocator provides buffer sub allocation using larger buffer blocks.
  The blocks are one VkBuffer each and are allocated via the
  provided [merian::MemAllocator](#class-merianmemallocator).

  The requested buffer space is sub-allocated and recycled in blocks internally.
  This way we avoid creating lots of small VkBuffers and can avoid calling the Vulkan
  API at all, when there are blocks with sufficient empty space.
  While Vulkan is more efficient than previous APIs, creating lots
  of objects for it, is still not good for overall performance. It will result
  into more cache misses and use more system memory over all.

  Be aware that each sub-allocation is always BASE_ALIGNMENT aligned.
  A custom alignment during allocation can be requested, it will ensure
  that the returned sub-allocation range of offset & size can account for
  the original requested size fitting within and respecting the requested

  This, however, means the regular offset and may not match the requested
  alignment, and the regular size can be bigger to account for the shift
  caused by manual alignment.

  It is therefore necessary to pass the alignment that was used at allocation time
  to the query functions as well.

  \code{.cpp}
  // alignment <= BASE_ALIGNMENT
      handle  = subAllocator.subAllocate(size);
      binding = subAllocator.getSubBinding(handle);

  // alignment > BASE_ALIGNMENT
      handle  = subAllocator.subAllocate(size, alignment);
      binding = subAllocator.getSubBinding(handle, alignment);
  \endcode
*/

class BufferSubAllocator {
  private:
    static const uint32_t INVALID_ID_INDEX = ~0;
    static const uint32_t BASE_ALIGNMENT = 16; // could compromise between max block size and typical requests

  public:
    class Handle {
        friend class BufferSubAllocator;

      private:
        static const uint32_t BLOCKBITS = 26;

        // if we cannot pack size and offset each into 26 bits (after adjusting for base alignment)
        // we need a dedicated block just for this
        static bool needsDedicated(uint64_t size, uint64_t alignment) {
            return ((size + (alignment > 16 ? alignment : 0)) >=
                    (uint64_t((1 << BLOCKBITS)) * uint64_t(BASE_ALIGNMENT)));
        }

        union {
            struct {
                uint64_t blockIndex : 11; // 2047 blocks, typical blockSize 64 MB or more, should be enough
                uint64_t offset : BLOCKBITS;
                uint64_t size : BLOCKBITS;
                uint64_t dedicated : 1; // 0 dedicated or not
            };
            uint64_t raw;
        };

        uint64_t getOffset() const {
            return dedicated == 1 ? 0 : offset * uint64_t(BASE_ALIGNMENT);
        }
        uint64_t getSize() const {
            return dedicated == 1 ? offset + (size << BLOCKBITS) : size * uint64_t(BASE_ALIGNMENT);
        }
        uint32_t getBlockIndex() const {
            return uint32_t(blockIndex);
        }
        bool isDedicated() const {
            return dedicated == 1;
        }

        bool setup(uint32_t blockIndex_, uint64_t offset_, uint64_t size_, bool dedicated_) {
            const uint64_t blockBitsMask = ((1ULL << BLOCKBITS) - 1);
            assert((blockIndex_ & ~((1ULL << 11) - 1)) == 0);
            blockIndex = blockIndex_ & ((1ULL << 11) - 1);
            if (dedicated_) {
                dedicated = 1;
                offset = size_ & blockBitsMask;
                size = (size_ >> BLOCKBITS) & blockBitsMask;
            } else {
                dedicated = 0;
                offset = (offset_ / uint64_t(BASE_ALIGNMENT)) & blockBitsMask;
                size = (size_ / uint64_t(BASE_ALIGNMENT)) & blockBitsMask;
            }

            return (getBlockIndex() == blockIndex_ && getOffset() == offset_ && getSize() == size_);
        }

      public:
        Handle() {
            raw = ~uint64_t(0);
        }

        bool isValid() const {
            return raw != ~uint64_t(0);
        }
        bool isEqual(const Handle& other) const {
            return blockIndex == other.blockIndex && offset == other.offset && dedicated == other.dedicated &&
                   size == other.size;
        }

        operator bool() const {
            return isValid();
        }

        friend bool operator==(const Handle& lhs, const Handle& rhs) {
            return rhs.isEqual(lhs);
        }
    };

    //////////////////////////////////////////////////////////////////////////
    BufferSubAllocator(BufferSubAllocator const&) = delete;
    BufferSubAllocator& operator=(BufferSubAllocator const&) = delete;

    BufferSubAllocator() {}
    BufferSubAllocator(MemoryAllocator* memAllocator,
                       vk::DeviceSize blockSize,
                       vk::BufferUsageFlags bufferUsageFlags,
                       vk::MemoryPropertyFlags memPropFlags = vk::MemoryPropertyFlagBits::eDeviceLocal,
                       bool mapped = false,
                       const std::vector<uint32_t>& sharingQueueFamilyIndices = std::vector<uint32_t>()) {
        init(memAllocator, blockSize, bufferUsageFlags, memPropFlags, mapped, sharingQueueFamilyIndices);
    }

    ~BufferSubAllocator() {
        deinit();
    }

    void init(MemoryAllocator* memallocator,
              vk::DeviceSize blockSize,
              vk::BufferUsageFlags bufferUsageFlags,
              vk::MemoryPropertyFlags memPropFlags = vk::MemoryPropertyFlagBits::eDeviceLocal,
              bool mapped = false,
              const std::vector<uint32_t>& sharingQueues = std::vector<uint32_t>());
    void deinit();

    void setKeepLastBlockOnFree(bool state) {
        m_keepLastBlock = state;
    }

    // alignment will be BASE_ALIGNMENT byte at least
    // alignment must be power of 2
    Handle subAllocate(vk::DeviceSize size, uint32_t alignment = BASE_ALIGNMENT);
    void subFree(Handle sub);

    struct Binding {
        vk::Buffer buffer;
        uint64_t offset;
        uint64_t size;
        vk::DeviceAddress address;
    };

    // sub allocation was aligned to BASE_ALIGNMENT
    Binding getSubBinding(Handle handle) {
        Binding binding;
        binding.offset = handle.getOffset();
        binding.size = handle.getSize();
        binding.buffer = m_blocks[handle.getBlockIndex()].buffer;
        binding.address = m_blocks[handle.getBlockIndex()].address + binding.offset;

        return binding;
    }
    // sub allocation alignment was custom
    Binding getSubBinding(Handle handle, uint32_t alignment) {
        Binding binding;
        binding.offset = (handle.getOffset() + (uint64_t(alignment) - 1)) & ~(uint64_t(alignment) - 1);
        binding.size = handle.getSize() - (binding.offset - handle.getOffset());
        binding.buffer = m_blocks[handle.getBlockIndex()].buffer;
        binding.address = m_blocks[handle.getBlockIndex()].address + binding.offset;

        return binding;
    }

    void* getSubMapping(Handle handle, uint32_t alignment = BASE_ALIGNMENT) const {
        return m_blocks[handle.getBlockIndex()].mapping +
               ((handle.getOffset() + (uint64_t(alignment) - 1)) & ~(uint64_t(alignment) - 1));
    }

    uint32_t getSubBlockIndex(Handle handle) const {
        return handle.getBlockIndex();
    }
    vk::Buffer getBlockBuffer(uint32_t blockIndex) const {
        return m_blocks[blockIndex].buffer;
    }

    float getUtilization(vk::DeviceSize& allocatedSize, vk::DeviceSize& usedSize) const;
    bool fitsInAllocated(vk::DeviceSize size, uint32_t alignment = BASE_ALIGNMENT) const;

    void free(bool onlyEmpty);

  protected:
    // - Block stores VkBuffers that we sub-allocate the staging space from

    // To recycle Block structures within the arrays
    // we use a linked list of array indices. The "index" element
    // in the struct refers to the next free list item, or itself
    // when in use.
    // A block is "dedicated" if it only holds a single allocation.
    // This can happen if we cannot encode the offset/size into the
    // bits that the Handle provides for this, or when the size
    // of the allocation is bigger than our preferred block size.

    struct Block {
        uint32_t index = INVALID_ID_INDEX;
        vk::DeviceSize size = 0;
        vk::Buffer buffer = VK_NULL_HANDLE;
        TRangeAllocator<BASE_ALIGNMENT> range;
        MemHandle memory = NullMemHandle;
        uint8_t* mapping = nullptr;
        vk::DeviceAddress address = 0;
        bool isDedicated = false;
    };

    MemoryAllocator* m_memAllocator = nullptr;
    vk::Device m_device = VK_NULL_HANDLE;
    uint32_t m_memoryTypeIndex;
    vk::DeviceSize m_blockSize;
    vk::BufferUsageFlags m_bufferUsageFlags;
    vk::MemoryPropertyFlags m_memoryPropFlags;
    std::vector<uint32_t> m_sharingQueueFamilyIndices;
    bool m_mapped;
    bool m_keepLastBlock = false;

    std::vector<Block> m_blocks;
    uint32_t m_regularBlocks = 0;
    uint32_t m_freeBlockIndex; // linked list to next free block
    vk::DeviceSize m_allocatedSize;
    vk::DeviceSize m_usedSize;

    uint32_t setIndexValue(uint32_t& index, uint32_t newValue) {
        uint32_t oldValue = index;
        index = newValue;
        return oldValue;
    }

    Block& getBlock(uint32_t index) {
        Block& block = m_blocks[index];
        assert(block.index == index);
        return block;
    }

    void freeBlock(Block& block);
    void allocBlock(Block& block, uint32_t id, vk::DeviceSize size);
};

} // namespace merian
