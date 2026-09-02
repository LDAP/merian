#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

/*
 * Builder for opacity micromaps. A micromap gives a triangle per-sub-triangle opacity states, so
 * that traversal can settle a hit against alpha-tested geometry without invoking a shader; only
 * the sub-triangles the micromap leaves unknown still reach the alpha test.
 *
 * A micromap is attached to a BLAS geometry through
 * vk::AccelerationStructureTrianglesOpacityMicromapEXT.
 */
class MicromapBuilder {

  public:
    // Device addresses of the build inputs must be a multiple of this.
    static constexpr vk::DeviceSize INPUT_ALIGNMENT = 256;

    MicromapBuilder(const ContextHandle& context, const ResourceAllocatorHandle& allocator);

    // Whether the device supports micromaps. Everything else here requires this to be true.
    static bool is_supported(const ContextHandle& context);

    // Queries the size, allocates a micromap and enqueues its build for the next get_cmds().
    //
    // `data` holds the packed opacity values and `triangle_array` one vk::MicromapTriangleEXT per
    // triangle, both at an offset that is a multiple of INPUT_ALIGNMENT. Both buffers must stay
    // alive until the command buffer completed.
    MicromapHandle queue_build(const vk::MicromapUsageEXT& usage,
                               const BufferHandle& data,
                               const vk::DeviceSize data_offset,
                               const BufferHandle& triangle_array,
                               const vk::DeviceSize triangle_array_offset,
                               const std::string& debug_name = {});

    // Records the enqueued builds and clears the queue. Returns false if nothing was enqueued.
    bool get_cmds(const CommandBufferHandle& cmd);

  private:
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;

    // pUsageCounts points into pending_usages, so the two stay index-aligned and are only
    // resolved in get_cmds(), after the vector stopped growing.
    std::vector<vk::MicromapUsageEXT> pending_usages;
    std::vector<vk::MicromapBuildInfoEXT> pending_build_infos;
    vk::DeviceSize pending_scratch_size = 0;

    vk::DeviceSize scratch_alignment;
};

} // namespace merian
