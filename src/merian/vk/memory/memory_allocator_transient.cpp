#include "merian/vk/memory/memory_allocator_transient.hpp"
#include "merian/utils/alignment.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <algorithm>
#include <bit>
#include <map>
#include <optional>
#include <spdlog/spdlog.h>

namespace merian {

TransientMemoryAllocator::Lifetime::Lifetime(const uint32_t positions, const bool all)
    : positions(positions), words((positions + 63) / 64, 0) {
    if (all) {
        add(0, positions);
    }
}

void TransientMemoryAllocator::Lifetime::add(const uint32_t first, const uint32_t count) {
    assert(count <= positions && (count == 0 || first < positions));
    for (uint32_t i = 0; i < count; i++) {
        const uint32_t position = (first + i) % positions;
        words[position / 64] |= uint64_t{1} << (position % 64);
    }
}

bool TransientMemoryAllocator::Lifetime::contains(const uint32_t position) const {
    assert(position < positions);
    return ((words[position / 64] >> (position % 64)) & 1u) != 0;
}

bool TransientMemoryAllocator::Lifetime::overlaps(const Lifetime& other) const {
    assert(positions == other.positions);
    for (uint32_t i = 0; i < words.size(); i++) {
        if ((words[i] & other.words[i]) != 0) {
            return true;
        }
    }
    return false;
}

uint32_t TransientMemoryAllocator::Lifetime::first() const {
    for (uint32_t i = 0; i < words.size(); i++) {
        if (words[i] != 0) {
            return 64 * i + std::countr_zero(words[i]);
        }
    }
    return positions;
}

uint32_t TransientMemoryAllocator::Lifetime::get_positions() const {
    return positions;
}

TransientMemoryAllocator::TransientMemoryAllocator(const MemoryAllocatorHandle& backing)
    : backing(backing) {}

TransientMemoryScopeHandle
TransientMemoryAllocator::create_scope(const Lifetime& lifetime,
                                       const MemoryAllocatorHandle& fallback) {
    return TransientMemoryScopeHandle(
        new TransientMemoryScope(shared_from_this(), fallback ? fallback : backing, lifetime));
}

uint32_t
TransientMemoryAllocator::add_request(const std::variant<ImageHandle, BufferHandle>& resource,
                                      const vk::MemoryRequirements& requirements,
                                      const uint32_t memory_type_index,
                                      const Lifetime& lifetime) {
    SPDLOG_DEBUG("transient request {}: {} (memory type {})", requests.size(),
                 format_size(requirements.size), memory_type_index);
    requests.emplace_back(resource, requirements, memory_type_index, lifetime);
    return static_cast<uint32_t>(requests.size() - 1);
}

void TransientMemoryAllocator::commit(std::vector<MemoryAllocationHandle> reusable) {
    assert(!committed);
    committed = true;

    // ordered: block indices are deterministic
    std::map<uint32_t, std::vector<uint32_t>> requests_for_memory_type;
    for (uint32_t i = 0; i < requests.size(); i++) {
        requests_for_memory_type[requests[i].memory_type_index].push_back(i);
    }

    // 1. the blocks and what they need
    placements.resize(requests.size());
    std::vector<BlockLayout> layouts;
    for (const auto& [memory_type_index, request_indices] : requests_for_memory_type) {
        for (std::vector<uint32_t>& block_request_indices :
             place(memory_type_index, request_indices, static_cast<uint32_t>(layouts.size()))) {
            layouts.push_back(layout_block(memory_type_index, std::move(block_request_indices)));
        }
    }

    // 2. the smallest reusable memory a block fits and fills at least half of
    std::vector<MemoryAllocationHandle> memory(layouts.size());
    for (uint32_t block_index = 0; block_index < layouts.size(); block_index++) {
        const BlockLayout& layout = layouts[block_index];
        auto reused = reusable.end();
        for (auto it = reusable.begin(); it != reusable.end(); it++) {
            const MemoryAllocationInfo info = (*it)->get_memory_info();
            if (info.memory_type_index == layout.memory_type_index &&
                info.offset % layout.alignment == 0 && layout.size <= info.size &&
                info.size <= 2 * layout.size &&
                (reused == reusable.end() || info.size < (*reused)->get_memory_info().size)) {
                reused = it;
            }
        }
        if (reused != reusable.end()) {
            memory[block_index] = *reused;
            reusable.erase(reused);
        }
    }
    reusable.clear();

    // 3. new memory for the others, bind
    for (uint32_t block_index = 0; block_index < layouts.size(); block_index++) {
        blocks.push_back(bind_block(layouts[block_index], memory[block_index]));
    }
    requests.clear();
}

std::vector<std::vector<uint32_t>>
TransientMemoryAllocator::place(const uint32_t memory_type_index,
                                std::vector<uint32_t> request_indices,
                                const uint32_t first_block_index) {
    const PhysicalDeviceHandle& physical_device = backing->get_context()->get_physical_device();
    const vk::DeviceSize granularity =
        physical_device->get_properties().get_properties().limits.bufferImageGranularity;
    const vk::PhysicalDeviceMemoryProperties& memory_properties =
        physical_device->get_memory_properties().memoryProperties;
    const vk::DeviceSize max_block_size = std::min(
        physical_device->get_properties().get_maintenance3_properties().maxMemoryAllocationSize,
        memory_properties.memoryHeaps[memory_properties.memoryTypes[memory_type_index].heapIndex]
            .size);

    // Greedy by size: Pisarchyk & Lee 2020, Efficient Memory Management for Deep Neural Net
    // Inference.
    std::sort(request_indices.begin(), request_indices.end(),
              [&](const uint32_t a, const uint32_t b) {
                  const Request& request_a = requests[a];
                  const Request& request_b = requests[b];
                  if (request_a.requirements.size != request_b.requirements.size) {
                      return request_a.requirements.size > request_b.requirements.size;
                  }
                  if (request_a.lifetime.first() != request_b.lifetime.first()) {
                      return request_a.lifetime.first() < request_b.lifetime.first();
                  }
                  return a < b;
              });

    // a request that fits no block below max_block_size opens the next one
    std::vector<std::vector<uint32_t>> block_request_indices;
    for (const uint32_t index : request_indices) {
        const vk::DeviceSize size = requests[index].requirements.size;
        const vk::DeviceSize alignment =
            std::max(requests[index].requirements.alignment, granularity);

        uint32_t block = 0;
        vk::DeviceSize offset = 0;
        for (; block < block_request_indices.size(); block++) {
            offset = find_offset(block_request_indices[block], index, alignment);
            if (offset + size <= max_block_size) {
                break;
            }
        }
        if (block == block_request_indices.size()) {
            block_request_indices.emplace_back();
            offset = 0;
        }

        block_request_indices[block].push_back(index);
        placements[index] = {.block_index = first_block_index + block,
                             .offset = offset,
                             .size = size,
                             .aliased = false};
        SPDLOG_DEBUG("transient request {}: {} in block {} at {}", index, format_size(size),
                     placements[index].block_index, format_size(offset));
    }

    return block_request_indices;
}

vk::DeviceSize TransientMemoryAllocator::find_offset(const std::vector<uint32_t>& placed,
                                                     const uint32_t request_index,
                                                     const vk::DeviceSize alignment) const {
    const Request& request = requests[request_index];
    std::vector<uint32_t> conflicts;
    std::copy_if(
        placed.begin(), placed.end(), std::back_inserter(conflicts),
        [&](const uint32_t other) { return requests[other].lifetime.overlaps(request.lifetime); });
    std::sort(conflicts.begin(), conflicts.end(), [&](const uint32_t a, const uint32_t b) {
        return placements[a].offset < placements[b].offset;
    });

    // the smallest gap that fits, else past the last conflict
    vk::DeviceSize end = 0;
    std::optional<vk::DeviceSize> best_offset;
    vk::DeviceSize best_gap = 0;
    for (const uint32_t other : conflicts) {
        const vk::DeviceSize offset = align_ceil(end, alignment);
        if (offset + request.requirements.size <= placements[other].offset &&
            (!best_offset || placements[other].offset - offset < best_gap)) {
            best_offset = offset;
            best_gap = placements[other].offset - offset;
        }
        end = std::max(end, placements[other].offset + placements[other].size);
    }
    return best_offset.value_or(align_ceil(end, alignment));
}

TransientMemoryAllocator::BlockLayout
TransientMemoryAllocator::layout_block(const uint32_t memory_type_index,
                                       std::vector<uint32_t> request_indices) {
    BlockLayout layout{.memory_type_index = memory_type_index,
                       .alignment = backing->get_context()
                                        ->get_physical_device()
                                        ->get_properties()
                                        .get_properties()
                                        .limits.bufferImageGranularity,
                       .request_indices = std::move(request_indices)};
    for (const uint32_t index : layout.request_indices) {
        Placement& placement = placements[index];
        layout.size = std::max(layout.size, placement.offset + placement.size);
        layout.requested_size += placement.size;
        layout.alignment = std::max(layout.alignment, requests[index].requirements.alignment);
        layout.memory_type_bits &= requests[index].requirements.memoryTypeBits;
        for (const uint32_t other : layout.request_indices) {
            placement.aliased |=
                other != index &&
                placement.offset < placements[other].offset + placements[other].size &&
                placements[other].offset < placement.offset + placement.size;
        }
    }
    return layout;
}

TransientMemoryAllocator::Block
TransientMemoryAllocator::bind_block(const BlockLayout& layout, MemoryAllocationHandle memory) {
    if (!memory) {
        const std::string debug_name = fmt::format("transient block {}", blocks.size());
        try {
            memory =
                backing->allocate_memory({},
                                         vk::MemoryRequirements{layout.size, layout.alignment,
                                                                1u << layout.memory_type_index},
                                         debug_name);
        } catch (const AllocationFailed&) {
            // the preferred type may be exhausted: fall back to any type all requests support
            memory = backing->allocate_memory(
                {}, vk::MemoryRequirements{layout.size, layout.alignment, layout.memory_type_bits},
                debug_name);
        }
    }

    for (const uint32_t index : layout.request_indices) {
        if (const ImageHandle* image = std::get_if<ImageHandle>(&requests[index].resource)) {
            memory->bind_to_image(*image, placements[index].offset);
        } else {
            memory->bind_to_buffer(std::get<BufferHandle>(requests[index].resource),
                                   placements[index].offset);
        }
    }

    const MemoryAllocationInfo memory_info = memory->get_memory_info();
    SPDLOG_DEBUG("transient block {} (memory type {}): {} resources, {} requested -> {}",
                 blocks.size(), memory_info.memory_type_index, layout.request_indices.size(),
                 format_size(layout.requested_size), format_size(memory_info.size));
    return {.memory_type_index = memory_info.memory_type_index,
            .size = memory_info.size,
            .requested_size = layout.requested_size,
            .memory = memory};
}

const std::vector<TransientMemoryAllocator::Block>& TransientMemoryAllocator::get_blocks() const {
    assert(committed);
    return blocks;
}

TransientMemoryAllocatorHandle
TransientMemoryAllocator::create(const MemoryAllocatorHandle& backing) {
    return TransientMemoryAllocatorHandle(new TransientMemoryAllocator(backing));
}

TransientMemoryScope::TransientMemoryScope(const TransientMemoryAllocatorHandle& allocator,
                                           const MemoryAllocatorHandle& fallback,
                                           const TransientMemoryAllocator::Lifetime& lifetime)
    : MemoryAllocator(allocator->backing->get_context()), allocator(allocator), fallback(fallback),
      lifetime(lifetime) {}

std::vector<TransientMemoryAllocator::Placement> TransientMemoryScope::get_placements() const {
    assert(allocator->committed);
    std::vector<TransientMemoryAllocator::Placement> placements;
    placements.reserve(placement_indices.size());
    for (const uint32_t index : placement_indices) {
        placements.push_back(allocator->placements[index]);
    }
    return placements;
}

std::vector<ImageHandle> TransientMemoryScope::get_aliased_images() const {
    assert(allocator->committed);
    std::vector<ImageHandle> aliased;
    for (uint32_t i = 0; i < placement_indices.size(); i++) {
        if (placed_images[i] && allocator->placements[placement_indices[i]].aliased) {
            aliased.push_back(placed_images[i]);
        }
    }
    return aliased;
}

const std::vector<TransientMemoryAllocator::Fallback>& TransientMemoryScope::get_fallbacks() const {
    return fallbacks;
}

const TransientMemoryAllocator::Lifetime& TransientMemoryScope::get_lifetime() const {
    return lifetime;
}

void TransientMemoryScope::check_not_committed() const {
    if (allocator->committed) {
        throw std::logic_error{"transient resource created after commit"};
    }
}

MemoryAllocationHandle
TransientMemoryScope::allocate_memory(const vk::MemoryPropertyFlags required_flags,
                                      const vk::MemoryRequirements& requirements,
                                      const std::string& debug_name,
                                      const MemoryMappingType mapping_type,
                                      const vk::MemoryPropertyFlags preferred_flags,
                                      const bool dedicated,
                                      const float dedicated_priority) {
    check_not_committed();
    fallbacks.emplace_back(requirements.size,
                           TransientMemoryAllocator::Fallback::Reason::RAW_MEMORY);
    return fallback->allocate_memory(required_flags, requirements, debug_name, mapping_type,
                                     preferred_flags, dedicated, dedicated_priority);
}

BufferHandle
TransientMemoryScope::create_buffer(const vk::BufferCreateInfo buffer_create_info,
                                    const MemoryMappingType mapping_type,
                                    const std::string& debug_name,
                                    const std::optional<vk::DeviceSize> min_alignment) {
    using Reason = TransientMemoryAllocator::Fallback::Reason;
    check_not_committed();

    constexpr vk::BufferCreateFlags own_memory_flags =
        vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency |
        vk::BufferCreateFlagBits::eSparseAliased |
        vk::BufferCreateFlagBits::eDeviceAddressCaptureReplay;
    std::optional<uint32_t> memory_type_index;
    std::optional<Reason> reason;
    if (mapping_type != MemoryMappingType::NONE) {
        reason = Reason::HOST_ACCESS;
    } else if (buffer_create_info.pNext != nullptr ||
               (buffer_create_info.flags & own_memory_flags)) {
        reason = Reason::OWN_MEMORY;
    } else if (memory_type_index = allocator->backing->find_memory_type(buffer_create_info);
               !memory_type_index) {
        reason = Reason::UNKNOWN_MEMORY_TYPE;
    }
    if (reason) {
        const BufferHandle buffer =
            fallback->create_buffer(buffer_create_info, mapping_type, debug_name, min_alignment);
        fallbacks.emplace_back(buffer->get_memory_requirements().size, *reason);
        return buffer;
    }

    const BufferHandle buffer = Buffer::create(get_context(), buffer_create_info);
    vk::MemoryRequirements requirements = buffer->get_memory_requirements();
    requirements.alignment = std::max(requirements.alignment, min_alignment.value_or(0));
    placement_indices.push_back(
        allocator->add_request(buffer, requirements, *memory_type_index, lifetime));
    placed_images.emplace_back();
    return buffer;
}

ImageHandle TransientMemoryScope::create_image(const vk::ImageCreateInfo image_create_info,
                                               const MemoryMappingType mapping_type,
                                               const std::string& debug_name) {
    using Reason = TransientMemoryAllocator::Fallback::Reason;
    check_not_committed();

    constexpr vk::ImageCreateFlags own_memory_flags =
        vk::ImageCreateFlagBits::eSparseBinding | vk::ImageCreateFlagBits::eSparseResidency |
        vk::ImageCreateFlagBits::eSparseAliased | vk::ImageCreateFlagBits::eDisjoint;
    std::optional<uint32_t> memory_type_index;
    std::optional<Reason> reason;
    if (mapping_type != MemoryMappingType::NONE) {
        reason = Reason::HOST_ACCESS;
    } else if (image_create_info.pNext != nullptr || (image_create_info.flags & own_memory_flags)) {
        reason = Reason::OWN_MEMORY;
    } else if (memory_type_index = allocator->backing->find_memory_type(image_create_info);
               !memory_type_index) {
        reason = Reason::UNKNOWN_MEMORY_TYPE;
    }
    if (reason) {
        const ImageHandle image =
            fallback->create_image(image_create_info, mapping_type, debug_name);
        fallbacks.emplace_back(image->get_memory_requirements().size, *reason);
        return image;
    }

    const ImageHandle image = Image::create(get_context(), image_create_info);
    placement_indices.push_back(allocator->add_request(image, image->get_memory_requirements(),
                                                       *memory_type_index, lifetime));
    placed_images.push_back(image);
    return image;
}

} // namespace merian
