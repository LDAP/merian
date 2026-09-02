#include "merian-shaders/scene/opacity_micromaps.hpp"

#include "merian/utils/properties.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace merian {

namespace {
constexpr uint32_t STATES_PER_WORD = 16;
// device addresses of the micromap build inputs must be a multiple of this
constexpr vk::DeviceSize INPUT_ALIGNMENT = MicromapBuilder::INPUT_ALIGNMENT;

// What one micromap may cost. The subdivision level is raised until the next one would exceed it,
// which is what keeps a micro-triangle down to a few texels without knowing the uv layout.
constexpr vk::DeviceSize MAX_BYTES_PER_MESH = 4u << 20;

// Micro-triangles one update may classify. Each costs at most OMM_MAX_TEXELS texel steps in the
// shader, so this bounds a submission: long enough to get through a scene in a few seconds, short
// enough that the driver never resets the device under it.
constexpr uint32_t MAX_MICRO_TRIANGLES_PER_UPDATE = 1u << 16;

uint32_t micro_triangles(const uint32_t subdivision_level) {
    return 1u << (2 * subdivision_level);
}

uint32_t words_per_triangle(const uint32_t subdivision_level) {
    return std::max(1u, micro_triangles(subdivision_level) / STATES_PER_WORD);
}

// Subdividing past the point where a micro-triangle covers a texel buys nothing, and a triangle
// covers at most the whole texture.
uint32_t texel_subdivision_limit(const vk::Extent2D texture_size) {
    const double edge =
        std::max(texture_size.width, texture_size.height) / double(MERIAN_OMM_TARGET_EDGE_TEXELS);
    if (!(edge > 1.0)) {
        return 0;
    }
    return static_cast<uint32_t>(std::ceil(std::log2(edge)));
}

// The finest subdivision that is still worth it and still fits the budget.
uint32_t choose_subdivision_level(const uint32_t primitive_count,
                                  const vk::Extent2D texture_size,
                                  const uint32_t max_level) {
    const uint32_t limit = std::min(max_level, texel_subdivision_limit(texture_size));
    uint32_t level = 0;
    for (uint32_t next = 1; next <= limit; next++) {
        if (vk::DeviceSize(primitive_count) * words_per_triangle(next) * sizeof(uint32_t) >
            MAX_BYTES_PER_MESH) {
            break;
        }
        level = next;
    }
    return level;
}

} // namespace

OpacityMicromaps::OpacityMicromaps(const ShaderCompileContextHandle& compile_context,
                                   const ContextHandle& context,
                                   const ResourceAllocatorHandle& allocator)
    : compile_context(compile_context), context(context), allocator(allocator),
      builder(context, allocator) {}

bool OpacityMicromaps::is_supported(const ContextHandle& context) {
    return MicromapBuilder::is_supported(context);
}

uint32_t OpacityMicromaps::max_subdivision_level() const {
    return context->get_physical_device()
        ->get_properties()
        .get_opacity_micromap_properties_ext()
        .maxOpacity4StateSubdivisionLevel;
}

void OpacityMicromaps::ensure_pipelines(const SlangCompositionHandle& material_system_composition) {
    if (composition) {
        return;
    }
    composition = SlangComposition::create();
    composition->add_composition(material_system_composition);
    composition->add_module_from_path("merian-shaders/scene/omm-bake.slang", true);
    program = SlangProgram::create(compile_context, composition);

    bake_entry_point = SlangProgramEntryPoint::create(program, "main");
    bake_pipeline = Versioned<Pipeline>([this] {
        const auto ep = bake_entry_point.get();
        return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
    });
    bake_pipeline.depends_on(bake_entry_point);
}

const ShaderObjectHandle&
OpacityMicromaps::next_object(std::vector<Versioned<ShaderObject>>& pool,
                              uint32_t& used,
                              const Versioned<SlangProgramEntryPoint>& entry_point) {
    if (used == pool.size()) {
        Versioned<ShaderObject> object([this, &entry_point] {
            return entry_point->create_shader_object_for_parameter(context, "params", allocator);
        });
        object.depends_on(entry_point);
        pool.emplace_back(std::move(object));
    }
    return pool[used++].get();
}

void OpacityMicromaps::update(const CommandBufferHandle& cmd,
                              const std::vector<MeshGeometry>& meshes,
                              const MaterialSystemHandle& material_system,
                              const uint32_t max_subdivision_level_setting,
                              const ShaderObjectAllocatorHandle& obj_allocator_in,
                              std::vector<uint32_t>& built) {
    const auto alpha_texture_of = [&](const MeshGeometry& mesh) {
        return uint32_t(material_system->get_alpha_texture_id(mesh.geometry.material_id));
    };
    const uint32_t max_level = std::min(max_subdivision_level_setting, max_subdivision_level());

    // 1. keep what is still valid, drop what is gone, collect what still needs baking
    std::unordered_map<uint32_t, Entry> kept;
    kept.reserve(entries.size());
    std::vector<MeshGeometry> pending;
    for (const MeshGeometry& mesh : meshes) {
        const auto it = entries.find(mesh.mesh_id);
        if (it != entries.end() && it->second.vertices == mesh.geometry.vertices &&
            it->second.indices == mesh.geometry.indices &&
            it->second.alpha_texture_id == alpha_texture_of(mesh) &&
            it->second.primitive_count == mesh.geometry.primitive_count) {
            kept.emplace(mesh.mesh_id, it->second);
            if (it->second.baked_triangles < it->second.primitive_count) {
                pending.push_back(mesh);
            }
            continue;
        }
        pending.push_back(mesh);
    }
    entries = std::move(kept);
    pending_count = static_cast<uint32_t>(pending.size());
    if (pending.empty()) {
        return;
    }

    MERIAN_PROFILE_SCOPE_GPU(cmd, "OpacityMicromaps::update");
    ensure_pipelines(material_system->get_composition());

    ShaderObjectAllocatorHandle obj_allocator = obj_allocator_in;
    if (!obj_allocator) {
        if (!fallback_obj_allocator) {
            fallback_obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);
        }
        obj_allocator = fallback_obj_allocator;
    }

    // 2. take a bounded slice of the outstanding triangles, resuming the meshes already started
    struct Slice {
        uint32_t mesh_id;
        uint32_t job_index;
        uint32_t first_record;
        uint32_t count;
    };
    std::vector<OmmBakeJob> jobs;
    std::vector<OmmBakeTriangle> records;
    std::vector<Slice> slices;
    uint32_t budget = MAX_MICRO_TRIANGLES_PER_UPDATE;

    for (const MeshGeometry& mesh : pending) {
        if (budget == 0) {
            break;
        }

        const auto [it, inserted] = entries.try_emplace(mesh.mesh_id);
        Entry& entry = it->second;
        if (inserted) {
            const vk::Extent3D extent = material_system->get_texture_manager()
                                            ->get_texture(alpha_texture_of(mesh))
                                            ->get_image()
                                            ->get_extent();
            const uint32_t level =
                choose_subdivision_level(mesh.geometry.primitive_count,
                                         vk::Extent2D{extent.width, extent.height}, max_level);
            entry.usage = vk::MicromapUsageEXT{mesh.geometry.primitive_count, level,
                                               MERIAN_OMM_FORMAT_4_STATE};
            entry.vertices = mesh.geometry.vertices;
            entry.indices = mesh.geometry.indices;
            entry.alpha_texture_id = alpha_texture_of(mesh);
            entry.primitive_count = mesh.geometry.primitive_count;
            entry.baked_triangles = 0;

            entry.data = allocator->create_buffer(
                vk::DeviceSize(entry.primitive_count) * words_per_triangle(level) *
                    sizeof(uint32_t),
                vk::BufferUsageFlagBits::eStorageBuffer |
                    vk::BufferUsageFlagBits::eShaderDeviceAddress |
                    vk::BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT,
                MemoryMappingType::NONE, "micromap data", INPUT_ALIGNMENT);

            // where the build finds each triangle's states; a fixed stride, so the host knows it
            std::vector<vk::MicromapTriangleEXT> array(entry.primitive_count);
            for (uint32_t i = 0; i < entry.primitive_count; i++) {
                array[i].dataOffset =
                    i * words_per_triangle(level) * static_cast<uint32_t>(sizeof(uint32_t));
                array[i].subdivisionLevel = static_cast<uint16_t>(level);
                array[i].format = MERIAN_OMM_FORMAT_4_STATE;
            }
            entry.triangles = allocator->create_buffer(
                cmd, array.size() * sizeof(vk::MicromapTriangleEXT),
                vk::BufferUsageFlagBits::eShaderDeviceAddress |
                    vk::BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT,
                array.data(), MemoryMappingType::NONE, "micromap triangles", INPUT_ALIGNMENT);
        }

        const uint32_t level = entry.usage.subdivisionLevel;
        const uint32_t per_triangle = micro_triangles(level);
        const uint32_t first = entry.baked_triangles;
        // at least one triangle, so a mesh subdivided past the budget still finishes eventually
        const uint32_t count =
            std::min(std::max(1u, budget / per_triangle), entry.primitive_count - first);

        OmmBakeJob job{};
        job.geometry = mesh.geometry;
        job.triangle_offset = first;
        job.primitive_count = entry.primitive_count;

        slices.emplace_back(Slice{mesh.mesh_id, static_cast<uint32_t>(jobs.size()),
                                  static_cast<uint32_t>(records.size()), count});
        jobs.emplace_back(job);
        for (uint32_t i = 0; i < count; i++) {
            OmmBakeTriangle record{};
            record.job_index = slices.back().job_index;
            record.primitive_id = first + i;
            record.subdivision_level = level;
            record.data_word_offset = (first + i) * words_per_triangle(level);
            records.emplace_back(record);
        }

        entry.baked_triangles = first + count;
        budget -= std::min(budget, count * per_triangle);
    }

    // 3. the shared index buffer: micromap triangle i for geometry triangle i
    uint32_t max_primitive_count = 0;
    for (const OmmBakeJob& job : jobs) {
        max_primitive_count = std::max(max_primitive_count, job.primitive_count);
    }
    if (!index_buffer ||
        index_buffer->get_size() < vk::DeviceSize(max_primitive_count) * sizeof(uint32_t)) {
        if (index_buffer) {
            cmd->keep_until_pool_reset(std::move(index_buffer));
        }
        std::vector<uint32_t> indices(max_primitive_count);
        std::iota(indices.begin(), indices.end(), 0u);
        index_buffer = allocator->create_buffer(
            cmd, indices.size() * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
                vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
            indices.data(), MemoryMappingType::NONE, "micromap triangle indices", INPUT_ALIGNMENT);
    }

    const BufferHandle job_buffer = allocator->create_buffer(
        cmd, jobs.size() * sizeof(OmmBakeJob),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        jobs.data(), MemoryMappingType::NONE, "micromap bake jobs");
    const BufferHandle record_buffer = allocator->create_buffer(
        cmd, records.size() * sizeof(OmmBakeTriangle),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        records.data(), MemoryMappingType::NONE, "micromap bake triangles");
    cmd->keep_until_pool_reset(job_buffer);
    cmd->keep_until_pool_reset(record_buffer);

    // 4. classify the slice, one workgroup per triangle and one dispatch per mesh
    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "bake");
        uint32_t used_bake_params = 0;
        const auto ep = bake_entry_point.get();
        const auto pipe = bake_pipeline.get();

        cmd->bind(pipe);
        ep->bind("material_system", material_system->get_shader_object(), cmd, pipe, obj_allocator);

        for (const Slice& slice : slices) {
            const ShaderObjectHandle& object =
                next_object(bake_params, used_bake_params, bake_entry_point);
            auto c = object->get_cursor();
            c["jobs"] = job_buffer;
            c["triangles"] = record_buffer;
            c["data"] = entries.at(slice.mesh_id).data;
            c["first_triangle"] = slice.first_record;
            c["triangle_count"] = slice.count;

            ep->bind("params", object, cmd, pipe, obj_allocator);
            cmd->dispatch(slice.count, 1, 1);
        }
    }

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eMicromapBuildEXT, vk::AccessFlagBits2::eMicromapReadEXT});

    // 5. build the micromap of every mesh whose last triangle was just baked
    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "build");
        bool any = false;
        for (const Slice& slice : slices) {
            Entry& entry = entries.at(slice.mesh_id);
            if (entry.baked_triangles < entry.primitive_count) {
                continue;
            }
            entry.micromap =
                builder.queue_build(entry.usage, entry.data, 0, entry.triangles, 0, "micromap");
            built.push_back(slice.mesh_id);
            any = true;
        }
        if (any) {
            builder.get_cmds(cmd);
        }
    }

    // the acceleration structure build reads what the micromap build just wrote
    cmd->barrier(vk::MemoryBarrier2{vk::PipelineStageFlagBits2::eMicromapBuildEXT,
                                    vk::AccessFlagBits2::eMicromapWriteEXT,
                                    vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                    vk::AccessFlagBits2::eMicromapReadEXT});

    data_size = 0;
    for (const auto& [mesh_id, entry] : entries) {
        data_size += entry.data ? entry.data->get_size() : 0;
    }
}

const MicromapHandle& OpacityMicromaps::get(const uint32_t mesh_id) const {
    static const MicromapHandle none;
    const auto it = entries.find(mesh_id);
    return it == entries.end() ? none : it->second.micromap;
}

const vk::MicromapUsageEXT& OpacityMicromaps::get_usage(const uint32_t mesh_id) const {
    static const vk::MicromapUsageEXT none;
    const auto it = entries.find(mesh_id);
    return it == entries.end() ? none : it->second.usage;
}

void OpacityMicromaps::clear() {
    entries.clear();
    data_size = 0;
    pending_count = 0;
}

void OpacityMicromaps::properties(Properties& props) {
    uint32_t complete = 0;
    for (const auto& [mesh_id, entry] : entries) {
        complete += entry.micromap ? 1 : 0;
    }
    props.output_text(fmt::format("micromaps: {}\nwaiting: {}\ndata: {}", complete, pending_count,
                                  format_size(data_size)));
}

} // namespace merian
