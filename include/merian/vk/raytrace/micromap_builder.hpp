#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

class MicromapBuilder {

  public:
    static constexpr vk::DeviceSize INPUT_ALIGNMENT = 256;

    MicromapBuilder(const ContextHandle& context, const ResourceAllocatorHandle& allocator);

    static bool is_supported(const ContextHandle& context);

    MicromapHandle queue_build(const vk::MicromapUsageEXT& usage,
                               const BufferHandle& data,
                               const vk::DeviceSize data_offset,
                               const BufferHandle& triangle_array,
                               const vk::DeviceSize triangle_array_offset,
                               const std::string& debug_name = {});

    bool get_cmds(const CommandBufferHandle& cmd);

  private:
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;

    struct PendingBuild {
        vk::MicromapBuildInfoEXT info;
        vk::MicromapUsageEXT usage;
        vk::DeviceSize scratch_size;
        BufferHandle data;
        BufferHandle triangle_array;
    };
    std::vector<PendingBuild> pending;

    vk::DeviceSize scratch_alignment;
};

} // namespace merian
