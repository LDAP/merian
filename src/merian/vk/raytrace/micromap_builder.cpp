#include "merian/vk/raytrace/micromap_builder.hpp"

#include "merian/utils/alignment.hpp"
#include "merian/vk/command/command_buffer.hpp"

namespace merian {

MicromapBuilder::MicromapBuilder(const ContextHandle& context,
                                 const ResourceAllocatorHandle& allocator)
    : context(context), allocator(allocator) {
    assert(is_supported(context));
    scratch_alignment = context->get_physical_device()
                            ->get_properties()
                            .get_acceleration_structure_properties_khr()
                            .minAccelerationStructureScratchOffsetAlignment;
}

bool MicromapBuilder::is_supported(const ContextHandle& context) {
    return context->get_device()
               ->get_enabled_features()
               .get_opacity_micromap_features_ext()
               .micromap == VK_TRUE;
}

MicromapHandle MicromapBuilder::queue_build(const vk::MicromapUsageEXT& usage,
                                            const BufferHandle& data,
                                            const vk::DeviceSize data_offset,
                                            const BufferHandle& triangle_array,
                                            const vk::DeviceSize triangle_array_offset,
                                            const std::string& debug_name) {
    const vk::DeviceAddress data_address = data->get_device_address() + data_offset;
    const vk::DeviceAddress triangle_array_address =
        triangle_array->get_device_address() + triangle_array_offset;
    assert(data_address % INPUT_ALIGNMENT == 0);
    assert(triangle_array_address % INPUT_ALIGNMENT == 0);

    vk::MicromapBuildInfoEXT build_info{vk::MicromapTypeEXT::eOpacityMicromap,
                                        vk::BuildMicromapFlagBitsEXT::ePreferFastTrace,
                                        vk::BuildMicromapModeEXT::eBuild,
                                        {},
                                        1,
                                        &usage,
                                        nullptr,
                                        vk::DeviceOrHostAddressConstKHR{data_address},
                                        {},
                                        vk::DeviceOrHostAddressConstKHR{triangle_array_address},
                                        sizeof(vk::MicromapTriangleEXT)};

    const vk::MicromapBuildSizesInfoEXT size_info =
        context->get_device()->get_device().getMicromapBuildSizesEXT(
            vk::AccelerationStructureBuildTypeKHR::eDevice, build_info);

    const MicromapHandle micromap =
        allocator->create_micromap(vk::MicromapTypeEXT::eOpacityMicromap, size_info, debug_name);
    build_info.dstMicromap = **micromap;

    pending.emplace_back(PendingBuild{build_info, usage,
                                      align_ceil(size_info.buildScratchSize, scratch_alignment),
                                      data, triangle_array});
    return micromap;
}

bool MicromapBuilder::get_cmds(const CommandBufferHandle& cmd) {
    if (pending.empty())
        return false;

    vk::DeviceSize scratch_size = 0;
    for (const PendingBuild& build : pending) {
        scratch_size += build.scratch_size;
    }
    const BufferHandle scratch = allocator->create_buffer(
        std::max<vk::DeviceSize>(scratch_size, 1),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        MemoryMappingType::NONE, "micromap scratch", scratch_alignment);

    std::vector<vk::MicromapBuildInfoEXT> build_infos;
    build_infos.reserve(pending.size());
    vk::DeviceAddress scratch_address = scratch->get_device_address();
    for (PendingBuild& build : pending) {
        build.info.pUsageCounts = &build.usage;
        build.info.scratchData = vk::DeviceOrHostAddressKHR{scratch_address};
        scratch_address += build.scratch_size;
        build_infos.emplace_back(build.info);
        cmd->keep_until_pool_reset(build.data);
        cmd->keep_until_pool_reset(build.triangle_array);
    }

    cmd->get_command_buffer().buildMicromapsEXT(build_infos);
    cmd->keep_until_pool_reset(scratch);
    pending.clear();

    return true;
}

} // namespace merian
