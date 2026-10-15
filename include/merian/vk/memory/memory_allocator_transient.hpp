#pragma once

#include "merian/vk/memory/memory_allocator.hpp"

#include <variant>

namespace merian {

class TransientMemoryAllocator;
using TransientMemoryAllocatorHandle = std::shared_ptr<TransientMemoryAllocator>;
class TransientMemoryScope;
using TransientMemoryScopeHandle = std::shared_ptr<TransientMemoryScope>;

// Places images and buffers that are not in use at the same time into shared memory blocks, which
// it allocates from the backing allocator.
//
// Resources are created through scopes, returned unbound and bound by commit(): create views and
// query device addresses only after that.
class TransientMemoryAllocator : public std::enable_shared_from_this<TransientMemoryAllocator> {
    friend TransientMemoryScope;

  public:
    static constexpr vk::DeviceSize MAX_BLOCK_SIZE = vk::DeviceSize{1} << 32;

    // The positions of a cycle at which a resource is in use.
    class Lifetime {
      public:
        // Of none or all positions.
        explicit Lifetime(const uint32_t positions, const bool all = false);

        // Adds count positions from first, wrapping around the end of the cycle.
        void add(const uint32_t first, const uint32_t count = 1);

        bool contains(const uint32_t position) const;

        bool overlaps(const Lifetime& other) const;

        // The lowest position in use, get_positions() if none.
        uint32_t first() const;

        uint32_t get_positions() const;

      private:
        uint32_t positions;
        std::vector<uint64_t> words;
    };

    struct Placement {
        uint32_t block_index;
        vk::DeviceSize offset;
        vk::DeviceSize size;
        // Another resource uses part of this memory while this one is not in use.
        bool aliased;
    };

    struct Block {
        uint32_t memory_type_index;
        // Size of the bound memory, at least the extent of the placements.
        vk::DeviceSize size;
        // Sum of the sizes of the resources placed in this block.
        vk::DeviceSize requested_size;
        MemoryAllocationHandle memory;
    };

    // A resource the fallback allocator serves.
    struct Fallback {
        enum class Reason {
            HOST_ACCESS,
            RAW_MEMORY,
            // sparse, disjoint, capture replay or extended (pNext)
            OWN_MEMORY,
            UNKNOWN_MEMORY_TYPE,
        };

        vk::DeviceSize size;
        Reason reason;
    };

  private:
    explicit TransientMemoryAllocator(const MemoryAllocatorHandle& backing);

  public:
    // Resources created through the scope are in use at the positions of the lifetime. Resources
    // that cannot be placed are allocated from fallback, or the backing allocator if null.
    TransientMemoryScopeHandle create_scope(const Lifetime& lifetime,
                                            const MemoryAllocatorHandle& fallback = nullptr);

    // Places and binds every resource. Call once, after all resources were created. A block reuses
    // memory it fits and fills at least half of; the rest is released before allocating new memory.
    void commit(std::vector<MemoryAllocationHandle> reusable = {});

    // Valid after commit().
    const std::vector<Block>& get_blocks() const;

  private:
    struct Request {
        std::variant<ImageHandle, BufferHandle> resource;
        vk::MemoryRequirements requirements;
        uint32_t memory_type_index;
        Lifetime lifetime;
    };

    // Returns the index of the placement.
    uint32_t add_request(const std::variant<ImageHandle, BufferHandle>& resource,
                         const vk::MemoryRequirements& requirements,
                         const uint32_t memory_type_index,
                         const Lifetime& lifetime);

    // Places the requests of one memory type into new blocks from first_block_index on, returns
    // the requests of each.
    std::vector<std::vector<uint32_t>> place(const uint32_t memory_type_index,
                                             std::vector<uint32_t> request_indices,
                                             const uint32_t first_block_index);

    // An offset at which the request overlaps no placed request it is in use with.
    vk::DeviceSize find_offset(const std::vector<uint32_t>& placed,
                               const uint32_t request_index,
                               const vk::DeviceSize alignment) const;

    struct BlockLayout {
        uint32_t memory_type_index;
        vk::DeviceSize alignment;
        vk::DeviceSize size = 0;
        vk::DeviceSize requested_size = 0;
        // every type in the intersection serves every request of the block
        uint32_t memory_type_bits = ~0u;
        std::vector<uint32_t> request_indices;
    };

    BlockLayout layout_block(const uint32_t memory_type_index,
                             std::vector<uint32_t> request_indices);

    // Binds the requests of a block to the memory, new memory if null.
    Block bind_block(const BlockLayout& layout, MemoryAllocationHandle memory);

    const MemoryAllocatorHandle backing;

    bool committed = false;
    // cleared on commit
    std::vector<Request> requests;
    // one per request (on commit)
    std::vector<Placement> placements;
    std::vector<Block> blocks;

  public:
    static TransientMemoryAllocatorHandle create(const MemoryAllocatorHandle& backing);
};

// Creates resources of one lifetime for a TransientMemoryAllocator. Mapped memory, raw allocations,
// sparse, disjoint, capture replay or extended (pNext) resources and resources whose memory type
// the backing allocator cannot tell are served by the fallback allocator.
class TransientMemoryScope : public MemoryAllocator {
    friend TransientMemoryAllocator;

  private:
    TransientMemoryScope(const TransientMemoryAllocatorHandle& allocator,
                         const MemoryAllocatorHandle& fallback,
                         const TransientMemoryAllocator::Lifetime& lifetime);

  public:
    // Of the resources placed through this scope, in creation order. Valid after commit().
    std::vector<TransientMemoryAllocator::Placement> get_placements() const;

    // The images placed through this scope that share memory with other resources. Valid after
    // commit().
    std::vector<ImageHandle> get_aliased_images() const;

    const std::vector<TransientMemoryAllocator::Fallback>& get_fallbacks() const;

    const TransientMemoryAllocator::Lifetime& get_lifetime() const;

    MemoryAllocationHandle
    allocate_memory(const vk::MemoryPropertyFlags required_flags,
                    const vk::MemoryRequirements& requirements,
                    const std::string& debug_name = {},
                    const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                    const vk::MemoryPropertyFlags preferred_flags = {},
                    const bool dedicated = false,
                    const float dedicated_priority = 1.0) override;

    BufferHandle
    create_buffer(const vk::BufferCreateInfo buffer_create_info,
                  const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                  const std::string& debug_name = {},
                  const std::optional<vk::DeviceSize> min_alignment = std::nullopt) override;

    ImageHandle create_image(const vk::ImageCreateInfo image_create_info,
                             const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                             const std::string& debug_name = {}) override;

  private:
    void check_not_committed() const;

    const TransientMemoryAllocatorHandle allocator;
    const MemoryAllocatorHandle fallback;
    const TransientMemoryAllocator::Lifetime lifetime;

    std::vector<uint32_t> placement_indices;
    // null for buffers
    std::vector<ImageHandle> placed_images;
    std::vector<TransientMemoryAllocator::Fallback> fallbacks;
};

} // namespace merian
