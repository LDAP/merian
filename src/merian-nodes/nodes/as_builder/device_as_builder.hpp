#pragma once

#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/connectors/vk_buffer_array_in.hpp"
#include "merian-nodes/connectors/vk_tlas_out.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/raytrace/as_builder.hpp"
#include "merian/vk/utils/math.hpp"

namespace merian_nodes {

/**
 * @brief      Builds BLASes and TLASes on the device.
 */
class DeviceASBuilder : public Node {
  public:
    /* Describe a BLAS to be build.
     * Once build, the BLAS is only rebuild if:
     *  - geometry is added or
     *  - request_rebuild() is called
     * Once build, the BLAS is only updated if:
     *  - the geometry did not change and
     *  - request_update() is called and
     *  - build_flags include allowUpdate
     *
     * Calling both request_update() and request_rebuild() results in a rebuild.
     */
    class BlasBuildInfo {
        friend DeviceASBuilder;

      public:
        BlasBuildInfo(const vk::BuildAccelerationStructureFlagsKHR build_flags)
            : build_flags(build_flags) {}

        /**
         * @brief      Adds geometry with rgb32f vertices and uint32 index.
         *
         * You must ensure that the buffers are not destructed until the build has finished.
         */
        BlasBuildInfo& add_geometry_f32_u32(const uint32_t vertex_count,
                                            const uint32_t primitive_count,
                                            const BufferHandle& vtx_buffer,
                                            const BufferHandle& idx_buffer) {
            // cannot update or reuse
            blas.reset();

            const vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
                vk::Format::eR32G32B32Sfloat,
                vtx_buffer->get_device_address(),
                3 * sizeof(float),
                vertex_count - 1, // why max and not count?!
                vk::IndexType::eUint32,
                idx_buffer->get_device_address(),
                {},
            };
            const vk::AccelerationStructureGeometryKHR geometry{
                vk::GeometryTypeKHR::eTriangles,
                vk::AccelerationStructureGeometryDataKHR{triangles}};
            const vk::AccelerationStructureBuildRangeInfoKHR range_info{primitive_count, 0, 0, 0};

            geometries.emplace_back(geometry);
            range_infos.emplace_back(range_info);

            return *this;
        }

        // call if you updated the geometry buffers and performed major deformations.
        void request_rebuild() {
            rebuild = true;
        }

        // call if you updated the geometry buffers and performed only slight deformations.
        void request_update() {
            update = true;
        }

      private:
        const vk::BuildAccelerationStructureFlagsKHR build_flags;

        // Info for the build
        std::vector<vk::AccelerationStructureGeometryKHR> geometries;
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> range_infos;

        // After the build stored here for rebuilds / updates
        AccelerationStructureHandle blas;
        bool update = false;
        bool rebuild = false;
    };

    /**
     * @brief      Describes a TLAS to be build.
     *
     * The TLAS is automatically rebuild if any of the BLASs is changed or a instance is added.
     * In the former case the previous tlas is overwritten in the latter a new tlas is constructed.
     *
     * TLASs are always rebuild and never updated (since that is not recommended anyways).
     */
    class TlasBuildInfo {
        friend DeviceASBuilder;

      public:
        TlasBuildInfo(const vk::BuildAccelerationStructureFlagsKHR build_flags =
                          vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
            : build_flags(build_flags) {}

        TlasBuildInfo&
        add_instance(const std::shared_ptr<BlasBuildInfo>& blas_info,
                     const vk::GeometryInstanceFlagsKHR instance_flags = {},
                     const uint32_t custom_index = 0,
                     const uint32_t mask = 0xFF,
                     const vk::TransformMatrixKHR& transform = merian::transform_identity()) {
            tlas.reset();

            // clang-format off
            const vk::AccelerationStructureInstanceKHR instance{
                transform,
                custom_index,
                mask,
                0,
                instance_flags,
                {}, // filled out later when the blas is actually built.
            };
            // clang-format on
            instances.emplace_back(instance);
            blases.emplace_back(blas_info);

            return *this;
        }

      private:
        const vk::BuildAccelerationStructureFlagsKHR build_flags;

        std::vector<vk::AccelerationStructureInstanceKHR> instances;
        std::vector<std::shared_ptr<BlasBuildInfo>> blases;

        // After the build stored here for rebuilds / updates
        AccelerationStructureHandle tlas;
        BufferHandle instances_buffer;

        bool rebuild = false;
    };

  private:
    struct InFlightData {
        std::vector<AccelerationStructureHandle> blases;
        BufferHandle instances_buffer;
        BufferHandle scratch_buffer;
    };

  public:
    DeviceASBuilder(const SharedContext& context, const ResourceAllocatorHandle& allocator)
        : Node("Acceleration Structure Builder"), as_builder(context, allocator) {}

    std::vector<InputConnectorHandle> describe_inputs() {
        return {
            con_in_instance_info,
            // con_in_vtx_buffers,
            // con_in_idx_buffers,
        };
    }

    std::vector<OutputConnectorHandle>
    describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
        return {
            con_out_tlas,
        };
    }

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) {
        TlasBuildInfo& tlas_build_info = *io[con_in_instance_info];

        // 1. Iterate over instances to queue the BLAS builds and update the BLAS addresses in the
        // instances
        for (uint32_t instance_index = 0; instance_index < tlas_build_info.instances.size();
             instance_index++) {
            BlasBuildInfo& blas_info = *tlas_build_info.blases[instance_index];
            vk::AccelerationStructureInstanceKHR& instance =
                tlas_build_info.instances[instance_index];

            if (!blas_info.blas) {
                // build
                blas_info.blas = as_builder.queue_build(blas_info.geometries, blas_info.range_infos,
                                                        blas_info.build_flags);
                tlas_build_info.rebuild = true;
            } else if (blas_info.rebuild) {
                // build reusing existing buffers and acceleration structures
                as_builder.queue_build(blas_info.geometries, blas_info.range_infos, blas_info.blas,
                                       blas_info.build_flags);
                tlas_build_info.rebuild = true;
            } else if (blas_info.update) {
                // update
                as_builder.queue_update(blas_info.geometries, blas_info.range_infos, blas_info.blas,
                                        blas_info.build_flags);
                tlas_build_info.rebuild = true;
            }

            blas_info.update = false;
            blas_info.rebuild = false;
            instance.accelerationStructureReference =
                blas_info.blas->get_acceleration_structure_device_address();
        }

        // 2. Create Instance buffer
        if (!allocator->ensureBufferSize(
                tlas_build_info.instances_buffer, size_of(tlas_build_info.instances),
                Buffer::INSTANCES_BUFFER_USAGE, "DeviceASBuilder Instances",
                merian::MemoryMappingType::NONE, 16, 1.25)) {
            // old buffer reused -> insert barrier for last iteration
            cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                vk::PipelineStageFlagBits::eTransfer, {}, {},
                tlas_build_info.instances_buffer->buffer_barrier(
                    vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferWrite),
                {});
        }
        // 2.1. Upload instances to GPU and copy to buffer
        allocator->getStaging()->cmdToBuffer(cmd, *tlas_build_info.instances_buffer, 0,
                                             size_of(tlas_build_info.instances),
                                             tlas_build_info.instances.data());
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, {},
            tlas_build_info.instances_buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                             vk::AccessFlagBits::eShaderRead),
            {});

        // 3. Queue TLAS build
        if (!tlas_build_info.tlas) {
            tlas_build_info.tlas = as_builder.queue_build(tlas_build_info.instances.size(),
                                   tlas_build_info.instances_buffer,
                                   tlas_build_info.build_flags);
        } else if (tlas_build_info.rebuild) {
            tlas_build_info.tlas = as_builder.queue_build(tlas_build_info.instances.size(),
                                                          tlas_build_info.instances_buffer,
                                                          tlas_build_info.build_flags);
        }

        // 4. Run builds (reusing the same scratch buffer)
        as_builder.get_cmds(cmd, scratch_buffer, run.get_profiler());

        // 5. Prevent object destruction
        InFlightData& in_flight_data = io.frame_data<InFlightData>();
        in_flight_data.scratch_buffer = scratch_buffer;
        in_flight_data.instances_buffer = tlas_build_info.instances_buffer;
        io[con_out_tlas] = tlas_build_info.tlas;
    }

  private:
    const ResourceAllocatorHandle allocator;
    ASBuilder as_builder;

    // clang-format off
    PtrInHandle<TlasBuildInfo> con_in_instance_info = PtrIn<TlasBuildInfo>::create("tlas_info");

    // todo: Add those as optional inputs somehow to ensure proper synchronization.
    // VkBufferArrayInHandle con_in_vtx_buffers = VkBufferArrayIn::acceleration_structure_read("vtx_buffers");
    // VkBufferArrayInHandle con_in_idx_buffers = VkBufferArrayIn::acceleration_structure_read("idx_buffers");

    VkTLASOutHandle con_out_tlas = VkTLASOut::create("tlas");
    // clang-format on

    BufferHandle scratch_buffer;
};

} // namespace merian_nodes
