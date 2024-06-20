#pragma once

#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/connectors/vk_tlas_out.hpp"
#include "merian-nodes/graph/node.hpp"
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
            blas.reset();
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
        TlasBuildInfo(const vk::BuildAccelerationStructureFlagsKHR build_flags)
            : build_flags(build_flags) {}

        TlasBuildInfo&
        add_instance(const std::shared_ptr<BlasBuildInfo>& blas,
                     const vk::GeometryInstanceFlagsKHR instance_flags = {},
                     const uint32_t custom_index = 0,
                     const uint32_t mask = 0xFF,
                     const vk::TransformMatrixKHR& transform = merian::transform_identity()) {
            tlas.reset();
            instances_buffer.reset();

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
            blases.emplace_back(blas);

            return *this;
        }

      private:
        const vk::BuildAccelerationStructureFlagsKHR build_flags;

        std::vector<vk::AccelerationStructureInstanceKHR> instances;
        std::vector<std::shared_ptr<BlasBuildInfo>> blases;

        // After the build stored here for rebuilds / updates
        AccelerationStructureHandle tlas;
        BufferHandle instances_buffer;
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
        };
    }

    std::vector<OutputConnectorHandle>
    describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
        return {
            con_out_tlas,
        };
    }

    void process([[maybe_unused]] GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) {
        InFlightData& in_flight_data = io.frame_data<InFlightData>();
        TlasBuildInfo& tlas_build_info = *io[con_in_instance_info];

        for (uint32_t instance_index = 0; instance_index < tlas_build_info.instances.size();
             instance_index++) {
        }


        if (tlas_build_info.tlas) {
            as_builder.queue_rebuild(tlas_build_info.instances.size(),
                                     in_flight_data.instances_buffer, tlas_build_info.tlas);
        } else {
            tlas_build_info.tlas = as_builder.queue_build(tlas_build_info.instances.size(),
                                                          in_flight_data.instances_buffer);
        }

        as_builder.get_cmds(cmd, in_flight_data.scratch_buffer);
        io[con_out_tlas] = tlas_build_info.tlas;
    }

  private:
    // clang-format off
    PtrInHandle<TlasBuildInfo> con_in_instance_info = PtrIn<TlasBuildInfo>::create("tlas_info");

    VkTLASOutHandle con_out_tlas = VkTLASOut::create("tlas");
    // clang-format on

    ASBuilder as_builder;
};

} // namespace merian_nodes
