#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/object.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

class ShaderBindingTableBuilder;
class ShaderBindingTable;
using ShaderBindingTableHandle = std::shared_ptr<ShaderBindingTable>;

// ---------------------------------------------------------------------------

/*
 * Shader Binding Table (SBT) for ray tracing dispatch.
 *
 * Inherits from Object so it can be kept alive via CommandBuffer::keep_until_pool_reset().
 *
 * Use ShaderBindingTable::create() for the common case (all pipeline groups in canonical
 * order, no per-record data).
 *
 * Use ShaderBindingTableBuilder for custom slot→group mapping and optional per-record
 * shader data (e.g. per-material records in hit groups).
 */
class ShaderBindingTable : public Object {
  public:
    // Default SBT: all pipeline groups in insertion order, no per-record data.
    // Region layout: raygen[0] | miss[0..n] | hit[0..n] | callable[0..n]
    //
    // Only the first raygen group is placed in the raygen region (Vulkan requires
    // raygen.stride == raygen.size, which limits the region to one entry per dispatch).
    static ShaderBindingTableHandle create(const RayTracingPipelineHandle& pipeline,
                                           const ResourceAllocatorHandle& allocator);

    friend class ShaderBindingTableBuilder;

    const vk::StridedDeviceAddressRegionKHR& get_raygen_region() const {
        return raygen_region;
    }
    const vk::StridedDeviceAddressRegionKHR& get_miss_region() const {
        return miss_region;
    }
    const vk::StridedDeviceAddressRegionKHR& get_hit_region() const {
        return hit_region;
    }
    const vk::StridedDeviceAddressRegionKHR& get_callable_region() const {
        return callable_region;
    }

  private:
    ShaderBindingTable() = default;

    BufferHandle sbt_buffer;
    vk::StridedDeviceAddressRegionKHR raygen_region{};
    vk::StridedDeviceAddressRegionKHR miss_region{};
    vk::StridedDeviceAddressRegionKHR hit_region{};
    vk::StridedDeviceAddressRegionKHR callable_region{};
};

// ---------------------------------------------------------------------------

/*
 * Builder for shader binding tables with custom slot→group mapping and optional
 * per-record shader data.
 *
 * All records in a region share the same stride. The stride is determined by
 * the first record that includes per-record data (subsequent records must use
 * the same data size, validated via assert).
 *
 * Example:
 *   struct MaterialRecord { uint32_t texture_id; float roughness; };
 *
 *   auto sbt = ShaderBindingTableBuilder(pipeline, allocator)
 *       .add_raygen(pipeline->group_index(raygen_ep))
 *       .add_miss(pipeline->group_index(miss_ep))
 *       .add_hit<MaterialRecord>(pipeline->group_index(chit_ep_a), {0, 0.5f})
 *       .add_hit<MaterialRecord>(pipeline->group_index(chit_ep_b), {1, 0.1f})
 *       .build();
 */
class ShaderBindingTableBuilder {
  public:
    ShaderBindingTableBuilder(const RayTracingPipelineHandle& pipeline,
                              const ResourceAllocatorHandle& allocator);

    // Add raygen entry. Only one raygen per SBT is supported (Vulkan requires
    // raygen.stride == raygen.size, which limits the region to a single entry).
    ShaderBindingTableBuilder& add_raygen(uint32_t group_index);

    ShaderBindingTableBuilder& add_miss(uint32_t group_index);

    template <typename T>
    ShaderBindingTableBuilder& add_miss(uint32_t group_index, const T& record_data) {
        return add_miss_raw(group_index, sizeof(T),
                            reinterpret_cast<const char*>(&record_data));
    }

    ShaderBindingTableBuilder& add_hit(uint32_t group_index);

    template <typename T>
    ShaderBindingTableBuilder& add_hit(uint32_t group_index, const T& record_data) {
        return add_hit_raw(group_index, sizeof(T),
                           reinterpret_cast<const char*>(&record_data));
    }

    ShaderBindingTableBuilder& add_callable(uint32_t group_index);

    template <typename T>
    ShaderBindingTableBuilder& add_callable(uint32_t group_index, const T& record_data) {
        return add_callable_raw(group_index, sizeof(T),
                                reinterpret_cast<const char*>(&record_data));
    }

    ShaderBindingTableHandle build();

  private:
    ShaderBindingTableBuilder& add_miss_raw(uint32_t group_index,
                                             uint32_t data_size,
                                             const char* data);
    ShaderBindingTableBuilder& add_hit_raw(uint32_t group_index,
                                            uint32_t data_size,
                                            const char* data);
    ShaderBindingTableBuilder& add_callable_raw(uint32_t group_index,
                                                 uint32_t data_size,
                                                 const char* data);

    struct Record {
        uint32_t          group_index;
        std::vector<char> data;
    };

    const RayTracingPipelineHandle pipeline;
    const ResourceAllocatorHandle  allocator;

    std::vector<Record> raygen_records;
    std::vector<Record> miss_records;
    std::vector<Record> hit_records;
    std::vector<Record> callable_records;

    // Per-region record data size (0 = no per-record data).
    // Set on first add_*_raw call with data; subsequent calls are validated against it.
    uint32_t miss_record_data_size     = 0;
    uint32_t hit_record_data_size      = 0;
    uint32_t callable_record_data_size = 0;
};

} // namespace merian
