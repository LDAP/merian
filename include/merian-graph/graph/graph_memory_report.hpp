#pragma once

#include "merian/vk/memory/memory_allocator_transient.hpp"

#include <string>
#include <vector>

namespace merian {

// Where a graph allocates memory: the outputs of its last connect and the memory the nodes
// allocated themselves that is alive.
struct GraphMemoryReport {
    struct Placed {
        // identifier
        std::string node;
        // with the copy and resource index where there is more than one
        std::string output;
        bool scratch;
        bool persistent;
        TransientMemoryAllocator::Placement placement;
        TransientMemoryAllocator::Lifetime lifetime;
    };

    struct Exclusive {
        std::string node;
        std::string output;
        bool scratch;
        TransientMemoryAllocator::Fallback fallback;
    };

    struct Block {
        uint32_t memory_type_index;
        vk::DeviceSize size;
        // of the placed resources
        vk::DeviceSize requested_size;
        std::weak_ptr<MemoryAllocation> memory;
    };

    struct NodeAllocation {
        std::string node;
        // with its exclusive outputs
        vk::DeviceSize size;
    };

    uint32_t layer_count{0};
    uint32_t period{1};
    bool aliasing{true};
    // The identifiers of the nodes of each layer.
    std::vector<std::vector<std::string>> layers;
    std::vector<Block> blocks;
    // One entry per placed resource, by block and offset.
    std::vector<Placed> placed;
    // Bytes placed in use at each position (iteration * layer_count + layer) of the period: no
    // placement fits into fewer than the maximum.
    std::vector<vk::DeviceSize> in_use;
    // Allocated by their node, largest first.
    std::vector<Exclusive> exclusive;
    // In run order, then the nodes that do not run.
    std::vector<NodeAllocation> node_allocations;

    std::string format_summary() const;
    std::string format_nodes() const;
    std::string format_layers() const;
    std::string format_outputs() const;
    std::string format_blocks() const;
};

std::string format_as(const GraphMemoryReport& report);

} // namespace merian
