#pragma once

#include "vk/extension/extension_vk_acceleration_structure.hpp"
#include "vk/memory/resource_allocator.hpp"

namespace merian {

/**
 * @brief      Front-end to create acceleration structure objects.
 *
 * To be efficient, ray tracing requires organizing the geometry into an acceleration structure (AS) that will reduce
 * the number of ray-triangle intersection tests during rendering. This is typically implemented in hardware as a
 * hierarchical structure, but only two levels are exposed to the user: a single top-level acceleration structure (TLAS)
 * referencing any number of bottom-level acceleration structures (BLAS), up to the limit
 * vk::PhysicalDeviceAccelerationStructurePropertiesKHR::maxInstanceCount. Typically, a BLAS corresponds to individual
 * 3D models within a scene, and a TLAS corresponds to an entire scene built by positioning (with 3-by-4 transformation
 * matrices) individual referenced BLASes.
 *
 * BLASes store the actual vertex data. They are built from one or more vertex buffers, each with its own transformation
 * matrix (separate from the TLAS matrices), allowing us to store multiple positioned models within a single BLAS. Note
 * that if an object is instantiated several times within the same BLAS, its geometry will be duplicated. This can be
 * particularly useful for improving performance on static, non-instantiated scene components (as a rule of thumb, the
 * fewer BLAS, the better).
 *
 * The TLAS will contain the object instances, each with its own transformation matrix and reference to a corresponding
 * BLAS. We will start with a single bottom-level AS and a top-level AS instancing it once with an identity transform.
 *
 * ~ quote from https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/
 *
 * This class acts as an owning container for a single top-level acceleration
 * structure referencing any number of bottom-level acceleration structures.
 * We provide functions for building (on the device) an array of BLASs and a
 * single TLAS from vectors of BlasInput and Instance, respectively, and
 * a destroy function for cleaning up the created acceleration structures.
 *
 * Generally, we reference BLASs by their index in the stored BLAS array,
 * rather than using raw device pointers as the pure Vulkan acceleration
 * structure API uses.
 *
 * This class does not support replacing acceleration structures once
 * built, but you can update the acceleration structures. For educational
 * purposes, this class prioritizes (relative) understandability over
 * performance, so vkQueueWaitIdle is implicitly used everywhere.
 *
 * # Setup and Usage
 * \code{.cpp}
 * // Borrow a VkDevice and memory allocator pointer (must remain
 * // valid throughout our use of the ray trace builder), and
 * // instantiate an unspecified queue of the given family for use.
 * rtBuilder = AccelerationStructureBuilder(...);
 *
 * // You create a vector of BlasInput then
 * // pass it to buildBlas.
 * std::vector<BlasInput> inputs = // ...
 * rtBuilder.buildBlas(inputs);
 *
 * // You create a vector of RaytracingBuilder::Instance and pass to
 * // buildTlas. The blasId member of each instance must be below
 * // inputs.size() (above).
 * std::vector<VkAccelerationStructureInstanceKHR> instances = // ...
 * rtBuilder.buildTlas(instances);
 *
 * // Retrieve the handle to the acceleration structure.
 * const VkAccelerationStructureKHR tlas = m.rtBuilder.getAccelerationStructure()
 * \endcode
 */
class AccelerationStructureBuilder {
  public:
    struct BlasInput {
        // Data used to build acceleration structure geometry
        std::vector<vk::AccelerationStructureGeometryKHR> asGeometry;
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> asBuildOffsetInfo;
        vk::BuildAccelerationStructureFlagsKHR flags;
    };

  public:
    AccelerationStructureBuilder(ExtensionVkAccelerationStructure& ext_acceleration_structure,
                                 vk::Device& device,
                                 ResourceAllocator& resource_allocator,
                                 QueueContainer& queue)
        : ext_acceleration_structure(ext_acceleration_structure), device(device),
          resource_allocator(resource_allocator), queue(queue) {

        if (!ext_acceleration_structure.is_supported()) {
            throw std::runtime_error{"Raytrace acceleration structure extension is not supported"};
        }
    }
    ~AccelerationStructureBuilder();

    // Returning the constructed top-level acceleration structure
    vk::AccelerationStructureKHR getAccelerationStructure() const;

    // Return the Acceleration Structure Device Address of a BLAS Id
    vk::DeviceAddress getBlasDeviceAddress(uint32_t blasId);

    // Create all the BLAS from the vector of BlasInput
    // - There will be one BLAS per input-vector entry
    // - There will be as many BLAS as input.size()
    // - The resulting BLAS (along with the inputs used to build) are stored in m_blas,
    //   and can be referenced by index.
    // - if flag has the 'Compact' flag, the BLAS will be compacted
    //
    void buildBLAS(
        const std::vector<BlasInput>& input,
        vk::BuildAccelerationStructureFlagsKHR flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Refit BLAS number blasIdx from updated buffer contents.
    void updateBLAS(uint32_t blasIdx, BlasInput& blas, vk::BuildAccelerationStructureFlagsKHR flags);

    // TODO: Static only?
    // Build TLAS from an array of vk::AccelerationStructureInstanceKHR
    // - The resulting TLAS will be stored internally and can be retrieved using getAccelerationStructure()
    // - update is to rebuild the Tlas with updated matrices, flag must have the 'allow_update'
    void buildTLAS(
        const std::vector<vk::AccelerationStructureInstanceKHR>& instances,
        vk::BuildAccelerationStructureFlagsKHR flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        bool update = false);

  protected:
    // Intermediate to hold build info of the AS and retrieve the result.
    struct BuildAccelerationStructureInfo {
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo;
        vk::AccelerationStructureBuildSizesInfoKHR sizeInfo;
        const vk::AccelerationStructureBuildRangeInfoKHR* rangeInfo;

        AccelKHR as;        // result acceleration structure
        AccelKHR cleanupAS; // used for old AS when compacting
    };

    void cmdCreateTLAS(vk::CommandBuffer& cmdBuf,                    // Command buffer
                       uint32_t countInstance,                       // number of instances
                       vk::DeviceAddress instBufferAddr,             // Buffer address of instances
                       Buffer& scratchBuffer,                        // Scratch buffer for construction
                       vk::BuildAccelerationStructureFlagsKHR flags, // Build creation flag
                       bool update                                   // Update == animation
    );
    void cmdCreateBLAS(vk::CommandBuffer& cmdBuf,
                       std::vector<uint32_t>& indices,
                       std::vector<BuildAccelerationStructureInfo>& buildAs,
                       vk::DeviceAddress scratchAddress,
                       std::optional<vk::QueryPool>& queryPool);
    void cmdCompactBLAS(vk::CommandBuffer& cmdBuf,
                        std::vector<uint32_t>& indices,
                        std::vector<BuildAccelerationStructureInfo>& buildAs,
                        vk::QueryPool& queryPool);
    void destroyNonCompactedBLAS(std::vector<uint32_t>& indices, std::vector<BuildAccelerationStructureInfo>& buildAs);

  private:
    ExtensionVkAccelerationStructure& ext_acceleration_structure;
    vk::Device& device;
    ResourceAllocator& resource_allocator;
    QueueContainer& queue;

    std::vector<AccelKHR> vec_blas; // Bottom-level acceleration structure
    AccelKHR tlas;                  // Top-level acceleration structure
};

} // namespace merian
