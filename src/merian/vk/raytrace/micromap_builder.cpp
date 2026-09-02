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

    pending_usages.emplace_back(usage);

    vk::MicromapBuildInfoEXT build_info{vk::MicromapTypeEXT::eOpacityMicromap,
                                        vk::BuildMicromapFlagBitsEXT::ePreferFastTrace,
                                        vk::BuildMicromapModeEXT::eBuild,
                                        {},
                                        1,
                                        &pending_usages.back(),
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
    pending_build_infos.emplace_back(build_info);
    pending_scratch_size += align_ceil(size_info.buildScratchSize, scratch_alignment);

    return micromap;
}

bool MicromapBuilder::get_cmds(const CommandBufferHandle& cmd) {
    if (pending_build_infos.empty())
        return false;

    const BufferHandle scratch = allocator->create_buffer(
        std::max<vk::DeviceSize>(pending_scratch_size, 1),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        MemoryMappingType::NONE, "micromap scratch", scratch_alignment);

    vk::DeviceAddress scratch_address = scratch->get_device_address();
    for (uint32_t i = 0; i < pending_build_infos.size(); i++) {
        pending_build_infos[i].pUsageCounts = &pending_usages[i];
        pending_build_infos[i].scratchData = vk::DeviceOrHostAddressKHR{scratch_address};
        const vk::MicromapBuildSizesInfoEXT size_info =
            context->get_device()->get_device().getMicromapBuildSizesEXT(
                vk::AccelerationStructureBuildTypeKHR::eDevice, pending_build_infos[i]);
        scratch_address += align_ceil(size_info.buildScratchSize, scratch_alignment);
    }

    cmd->get_command_buffer().buildMicromapsEXT(pending_build_infos);
    cmd->keep_until_pool_reset(scratch);

    pending_build_infos.clear();
    pending_usages.clear();
    pending_scratch_size = 0;

    return true;
}

} // namespace merian
