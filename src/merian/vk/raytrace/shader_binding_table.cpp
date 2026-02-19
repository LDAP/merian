#include "merian/vk/raytrace/shader_binding_table.hpp"
#include "merian/vk/memory/memory_allocator.hpp"

#include <cstring>

namespace merian {

namespace {

inline vk::DeviceSize align_up(vk::DeviceSize val, vk::DeviceSize align) {
    if (align == 0)
        return val;
    return ((val + align - 1) / align) * align;
}

} // namespace

// ---------------------------------------------------------------------------
// ShaderBindingTable::create — default SBT, no per-record data
// ---------------------------------------------------------------------------

ShaderBindingTableHandle ShaderBindingTable::create(const RayTracingPipelineHandle& pipeline,
                                                    const ResourceAllocatorHandle& allocator) {
    const auto& ctx = pipeline->get_context();
    const auto& rt_props =
        ctx->get_physical_device()->get_properties().get_ray_tracing_pipeline_properties_khr();

    const vk::DeviceSize handle_size      = rt_props.shaderGroupHandleSize;
    const vk::DeviceSize handle_alignment = rt_props.shaderGroupHandleAlignment;
    const vk::DeviceSize base_alignment   = rt_props.shaderGroupBaseAlignment;

    // Stride for one record with no per-record data
    const vk::DeviceSize record_stride = align_up(handle_size, handle_alignment);

    const vk::DeviceSize n_raygen   = pipeline->get_raygen_count();
    const vk::DeviceSize n_miss     = pipeline->get_miss_count();
    const vk::DeviceSize n_hit      = pipeline->get_hit_count();
    const vk::DeviceSize n_callable = pipeline->get_callable_count();

    // Compute region offsets — each region starts at a base_alignment boundary.
    // Only the FIRST raygen group is placed in the raygen region because
    // Vulkan requires raygen.stride == raygen.size (single entry per dispatch).
    vk::DeviceSize offset = 0;

    const vk::DeviceSize raygen_region_bytes = (n_raygen > 0) ? record_stride : 0;
    const vk::DeviceSize raygen_offset       = offset;
    offset += raygen_region_bytes;

    offset                                 = align_up(offset, base_alignment);
    const vk::DeviceSize miss_region_bytes = n_miss * record_stride;
    const vk::DeviceSize miss_offset       = offset;
    offset += miss_region_bytes;

    offset                                = align_up(offset, base_alignment);
    const vk::DeviceSize hit_region_bytes = n_hit * record_stride;
    const vk::DeviceSize hit_offset       = offset;
    offset += hit_region_bytes;

    offset                                     = align_up(offset, base_alignment);
    const vk::DeviceSize callable_region_bytes = n_callable * record_stride;
    const vk::DeviceSize callable_offset       = offset;
    offset += callable_region_bytes;

    const vk::DeviceSize buffer_size = offset;

    auto sbt = ShaderBindingTableHandle(new ShaderBindingTable());

    if (buffer_size == 0)
        return sbt;

    // Fetch all shader group handles from the pipeline (canonical order:
    // raygen[0..n_raygen-1] | miss[0..n_miss-1] | hit[0..n_hit-1] | callable[0..n_callable-1])
    const std::vector<uint8_t> all_handles = pipeline->get_shader_group_handles();

    const auto usage = vk::BufferUsageFlagBits::eShaderBindingTableKHR |
                       vk::BufferUsageFlagBits::eShaderDeviceAddress;
    sbt->sbt_buffer = allocator->create_buffer(buffer_size, usage,
                                                MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE,
                                                "ShaderBindingTable");

    auto* data = sbt->sbt_buffer->get_memory()->map_as<uint8_t>();
    std::memset(data, 0, static_cast<size_t>(buffer_size));

    // Write raygen (first entry only — group 0 = raygen[0])
    if (n_raygen > 0) {
        std::memcpy(data + raygen_offset, all_handles.data(), static_cast<size_t>(handle_size));
    }

    // Write miss
    for (vk::DeviceSize i = 0; i < n_miss; ++i) {
        const vk::DeviceSize abs_idx = n_raygen + i;
        const uint8_t* src           = all_handles.data() + (abs_idx * handle_size);
        std::memcpy(data + miss_offset + (i * record_stride), src,
                    static_cast<size_t>(handle_size));
    }

    // Write hit (triangle groups then procedural groups, in pipeline insertion order)
    for (vk::DeviceSize i = 0; i < n_hit; ++i) {
        const vk::DeviceSize abs_idx = n_raygen + n_miss + i;
        const uint8_t* src           = all_handles.data() + (abs_idx * handle_size);
        std::memcpy(data + hit_offset + (i * record_stride), src,
                    static_cast<size_t>(handle_size));
    }

    // Write callable
    for (vk::DeviceSize i = 0; i < n_callable; ++i) {
        const vk::DeviceSize abs_idx = n_raygen + n_miss + n_hit + i;
        const uint8_t* src           = all_handles.data() + (abs_idx * handle_size);
        std::memcpy(data + callable_offset + (i * record_stride), src,
                    static_cast<size_t>(handle_size));
    }

    sbt->sbt_buffer->get_memory()->unmap();

    const vk::DeviceAddress buf_addr = sbt->sbt_buffer->get_device_address();

    if (n_raygen > 0)
        sbt->raygen_region = vk::StridedDeviceAddressRegionKHR{
            buf_addr + raygen_offset, record_stride, record_stride};

    if (n_miss > 0)
        sbt->miss_region = vk::StridedDeviceAddressRegionKHR{
            buf_addr + miss_offset, record_stride, miss_region_bytes};

    if (n_hit > 0)
        sbt->hit_region = vk::StridedDeviceAddressRegionKHR{
            buf_addr + hit_offset, record_stride, hit_region_bytes};

    if (n_callable > 0)
        sbt->callable_region = vk::StridedDeviceAddressRegionKHR{
            buf_addr + callable_offset, record_stride, callable_region_bytes};

    return sbt;
}

// ---------------------------------------------------------------------------
// ShaderBindingTableBuilder
// ---------------------------------------------------------------------------

ShaderBindingTableBuilder::ShaderBindingTableBuilder(const RayTracingPipelineHandle& pipeline,
                                                     const ResourceAllocatorHandle& allocator)
    : pipeline(pipeline), allocator(allocator) {}

ShaderBindingTableBuilder& ShaderBindingTableBuilder::add_raygen(uint32_t group_index) {
    assert(raygen_records.empty() && "only one raygen record per SBT is supported");
    raygen_records.push_back({group_index, {}});
    return *this;
}

ShaderBindingTableBuilder& ShaderBindingTableBuilder::add_miss(uint32_t group_index) {
    miss_records.push_back({group_index, {}});
    return *this;
}

ShaderBindingTableBuilder&
ShaderBindingTableBuilder::add_miss_raw(uint32_t group_index,
                                        uint32_t data_size,
                                        const char* data) {
    if (miss_record_data_size == 0)
        miss_record_data_size = data_size;
    assert(miss_record_data_size == data_size &&
           "all miss records must have the same per-record data size");
    miss_records.push_back({group_index, std::vector<char>(data, data + data_size)});
    return *this;
}

ShaderBindingTableBuilder& ShaderBindingTableBuilder::add_hit(uint32_t group_index) {
    hit_records.push_back({group_index, {}});
    return *this;
}

ShaderBindingTableBuilder&
ShaderBindingTableBuilder::add_hit_raw(uint32_t group_index,
                                       uint32_t data_size,
                                       const char* data) {
    if (hit_record_data_size == 0)
        hit_record_data_size = data_size;
    assert(hit_record_data_size == data_size &&
           "all hit records must have the same per-record data size");
    hit_records.push_back({group_index, std::vector<char>(data, data + data_size)});
    return *this;
}

ShaderBindingTableBuilder& ShaderBindingTableBuilder::add_callable(uint32_t group_index) {
    callable_records.push_back({group_index, {}});
    return *this;
}

ShaderBindingTableBuilder&
ShaderBindingTableBuilder::add_callable_raw(uint32_t group_index,
                                             uint32_t data_size,
                                             const char* data) {
    if (callable_record_data_size == 0)
        callable_record_data_size = data_size;
    assert(callable_record_data_size == data_size &&
           "all callable records must have the same per-record data size");
    callable_records.push_back({group_index, std::vector<char>(data, data + data_size)});
    return *this;
}

ShaderBindingTableHandle ShaderBindingTableBuilder::build() {
    const auto& ctx = pipeline->get_context();
    const auto& rt_props =
        ctx->get_physical_device()->get_properties().get_ray_tracing_pipeline_properties_khr();

    const vk::DeviceSize handle_size      = rt_props.shaderGroupHandleSize;
    const vk::DeviceSize handle_alignment = rt_props.shaderGroupHandleAlignment;
    const vk::DeviceSize base_alignment   = rt_props.shaderGroupBaseAlignment;

    // Per-region strides (handle + optional per-record data, aligned)
    const vk::DeviceSize raygen_stride = align_up(handle_size, handle_alignment);

    const vk::DeviceSize miss_stride =
        miss_records.empty()
            ? 0
            : align_up(handle_size + miss_record_data_size, handle_alignment);

    const vk::DeviceSize hit_stride =
        hit_records.empty()
            ? 0
            : align_up(handle_size + hit_record_data_size, handle_alignment);

    const vk::DeviceSize callable_stride =
        callable_records.empty()
            ? 0
            : align_up(handle_size + callable_record_data_size, handle_alignment);

    // Compute region sizes and offsets
    vk::DeviceSize offset = 0;

    // Raygen: stride == size (single entry, Vulkan spec requirement)
    const vk::DeviceSize raygen_region_bytes = raygen_records.empty() ? 0 : raygen_stride;
    const vk::DeviceSize raygen_offset       = offset;
    offset += raygen_region_bytes;

    offset                                 = align_up(offset, base_alignment);
    const vk::DeviceSize miss_region_bytes = miss_records.size() * miss_stride;
    const vk::DeviceSize miss_offset       = offset;
    offset += miss_region_bytes;

    offset                                = align_up(offset, base_alignment);
    const vk::DeviceSize hit_region_bytes = hit_records.size() * hit_stride;
    const vk::DeviceSize hit_offset       = offset;
    offset += hit_region_bytes;

    offset                                     = align_up(offset, base_alignment);
    const vk::DeviceSize callable_region_bytes = callable_records.size() * callable_stride;
    const vk::DeviceSize callable_offset       = offset;
    offset += callable_region_bytes;

    const vk::DeviceSize buffer_size = offset;

    auto sbt = ShaderBindingTableHandle(new ShaderBindingTable());

    if (buffer_size == 0)
        return sbt;

    const std::vector<uint8_t> all_handles = pipeline->get_shader_group_handles();

    const auto usage = vk::BufferUsageFlagBits::eShaderBindingTableKHR |
                       vk::BufferUsageFlagBits::eShaderDeviceAddress;
    sbt->sbt_buffer = allocator->create_buffer(buffer_size, usage,
                                                MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE,
                                                "ShaderBindingTable");

    auto* data = sbt->sbt_buffer->get_memory()->map_as<uint8_t>();
    std::memset(data, 0, static_cast<size_t>(buffer_size));

    const auto write_record = [&](uint8_t* dst, const Record& rec) {
        const uint8_t* handle_src =
            all_handles.data() + (static_cast<vk::DeviceSize>(rec.group_index) * handle_size);
        std::memcpy(dst, handle_src, static_cast<size_t>(handle_size));
        if (!rec.data.empty())
            std::memcpy(dst + handle_size, rec.data.data(), rec.data.size());
    };

    // Write raygen (one entry)
    if (!raygen_records.empty())
        write_record(data + raygen_offset, raygen_records[0]);

    // Write miss
    for (size_t i = 0; i < miss_records.size(); ++i)
        write_record(data + miss_offset + (i * miss_stride), miss_records[i]);

    // Write hit
    for (size_t i = 0; i < hit_records.size(); ++i)
        write_record(data + hit_offset + (i * hit_stride), hit_records[i]);

    // Write callable
    for (size_t i = 0; i < callable_records.size(); ++i)
        write_record(data + callable_offset + (i * callable_stride), callable_records[i]);

    sbt->sbt_buffer->get_memory()->unmap();

    const vk::DeviceAddress buf_addr = sbt->sbt_buffer->get_device_address();

    if (!raygen_records.empty())
        sbt->raygen_region = vk::StridedDeviceAddressRegionKHR{
            buf_addr + raygen_offset, raygen_stride, raygen_stride};

    if (!miss_records.empty())
        sbt->miss_region = vk::StridedDeviceAddressRegionKHR{
            buf_addr + miss_offset, miss_stride, miss_region_bytes};

    if (!hit_records.empty())
        sbt->hit_region = vk::StridedDeviceAddressRegionKHR{
            buf_addr + hit_offset, hit_stride, hit_region_bytes};

    if (!callable_records.empty())
        sbt->callable_region = vk::StridedDeviceAddressRegionKHR{
            buf_addr + callable_offset, callable_stride, callable_region_bytes};

    return sbt;
}

} // namespace merian
