#include "merian-shaders/scene/opacity_micromaps.hpp"

#include "merian/utils/properties.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>

namespace merian {

namespace {
constexpr uint32_t STATES_PER_WORD = 16;
// device addresses of the micromap build inputs must be a multiple of this
constexpr vk::DeviceSize INPUT_ALIGNMENT = MicromapBuilder::INPUT_ALIGNMENT;

// What one triangle's states may cost. Traversal reads this for every alpha-tested hit, and past a
// point the traffic costs more than the alpha tests a finer subdivision saves: on Emerald Square
// 64 bytes (level 4) renders in 44.59 ms against 51.46 at 1024 bytes (level 6), even though the
// coarser one leaves twice as many micro-triangles unknown.
constexpr vk::DeviceSize MAX_BYTES_PER_TRIANGLE = 64;

// Texel steps one update may take, counted at the worst case a triangle can reach. Bounding the
// work rather than the micro-triangle count keeps a submission the same length whatever
// subdivision level the meshes end up at; the measured cost is around a tenth of the bound.
constexpr uint64_t MAX_TEXEL_STEPS_PER_UPDATE = 400'000'000;

uint32_t micro_triangles(const uint32_t subdivision_level) {
    return 1u << (2 * subdivision_level);
}

uint32_t words_per_triangle(const uint32_t subdivision_level) {
    return std::max(1u, micro_triangles(subdivision_level) / STATES_PER_WORD);
}

// What rasterizing one triangle's micro-triangles can cost at this level, in texel steps.
uint64_t bake_cost(const uint32_t level, const vk::Extent2D texture_size) {
    const uint64_t span =
        (uint64_t(std::max(texture_size.width, texture_size.height)) >> level) + 3;
    return uint64_t(micro_triangles(level)) *
           std::min<uint64_t>(MERIAN_OMM_MAX_TEXELS, span * span);
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
uint32_t choose_subdivision_level(const vk::Extent2D texture_size, const uint32_t max_level) {
    const uint32_t limit = std::min(max_level, texel_subdivision_limit(texture_size));
    uint32_t level = 0;
    for (uint32_t next = 1; next <= limit; next++) {
        if (words_per_triangle(next) * sizeof(uint32_t) > MAX_BYTES_PER_TRIANGLE) {
            break;
        }
        level = next;
    }
    return level;
}

// Interleaves the low 16 bits of each axis, so blocks that are close in the texture end up close
// in the buffer.
uint32_t morton(const uint32_t x, const uint32_t y) {
    const auto spread = [](uint32_t v) {
        v &= 0xFFFF;
        v = (v | (v << 8)) & 0x00FF00FF;
        v = (v | (v << 4)) & 0x0F0F0F0F;
        v = (v | (v << 2)) & 0x33333333;
        v = (v | (v << 1)) & 0x55555555;
        return v;
    };
    return spread(x) | (spread(y) << 1);
}

// Triangles whose texture coordinates are bit-identical produce the same states, so they can share
// one block. Foliage instances its cards heavily, and the coordinates are stored at half precision,
// which folds the nearly-identical ones together as well: Emerald Square's 1.5 M alpha-tested
// triangles come down to some 2700 blocks. Following NVIDIA's Opacity Micro-Map SDK, which calls
// this the reuse pre-pass.
uint32_t build_blocks(const OpacityMicromaps::MeshGeometry& mesh,
                      std::vector<uint32_t>& block_of_triangle,
                      std::vector<uint32_t>& representative) {
    const uint32_t count = mesh.geometry.primitive_count;
    block_of_triangle.resize(count);
    representative.clear();

    if (true || mesh.host_vertices == nullptr) { // DIAGNOSTIC: dedup off
        // nothing to compare on the host: every triangle keeps its own block
        std::iota(block_of_triangle.begin(), block_of_triangle.end(), 0u);
        representative.resize(count);
        std::iota(representative.begin(), representative.end(), 0u);
        return count;
    }

    const auto corner = [&](const uint32_t triangle, const uint32_t i) -> uint32_t {
        uint32_t vertex = 3 * triangle + i;
        if (mesh.host_indices != nullptr) {
            vertex = mesh.host_index_type == vk::IndexType::eUint16
                         ? static_cast<const uint16_t*>(mesh.host_indices)[vertex]
                         : static_cast<const uint32_t*>(mesh.host_indices)[vertex];
        }
        // the uv pair sits in one dword, so its bits are the key
        uint32_t bits;
        std::memcpy(&bits, &mesh.host_vertices[vertex].uv, sizeof(bits));
        return bits;
    };

    // the coordinates themselves are the key, never a digest of them: two triangles may only share
    // a block when their states are certain to match
    struct Key {
        std::array<uint32_t, 3> uv;
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            return size_t(k.uv[0] * 0x9E3779B9ull) ^ size_t(k.uv[1] * 0xC2B2AE3Dull) ^
                   size_t(k.uv[2] * 0x165667B1ull);
        }
    };

    std::unordered_map<Key, uint32_t, KeyHash> seen;
    seen.reserve(count / 4 + 1);
    for (uint32_t triangle = 0; triangle < count; triangle++) {
        const Key key{{corner(triangle, 0), corner(triangle, 1), corner(triangle, 2)}};
        const auto [it, inserted] = seen.try_emplace(key, static_cast<uint32_t>(seen.size()));
        block_of_triangle[triangle] = it->second;
        if (inserted) {
            representative.push_back(triangle);
        }
    }

    // Traversal reads a block for every alpha-tested hit, and rays that are near each other hit
    // triangles that are near each other in the texture. Ordering the blocks along a curve over
    // that domain is what keeps those reads together; sharing blocks scatters them otherwise.
    const uint32_t blocks = static_cast<uint32_t>(representative.size());
    std::vector<uint32_t> order(blocks);
    std::iota(order.begin(), order.end(), 0u);
    std::vector<uint32_t> code(blocks);
    for (uint32_t block = 0; block < blocks; block++) {
        const uint32_t triangle = representative[block];
        float u = 0.f;
        float v = 0.f;
        for (uint32_t i = 0; i < 3; i++) {
            const uint32_t bits = corner(triangle, i);
            half2 uv;
            std::memcpy(&uv, &bits, sizeof(bits));
            u += float(uv.x);
            v += float(uv.y);
        }
        const auto quantize = [](const float c) {
            return static_cast<uint32_t>(std::clamp(c / 3.f, 0.f, 1.f) * 65535.f);
        };
        code[block] = morton(quantize(u), quantize(v));
    }
    std::sort(order.begin(), order.end(),
              [&](const uint32_t a, const uint32_t b) { return code[a] < code[b]; });

    std::vector<uint32_t> remap(blocks);
    std::vector<uint32_t> sorted(blocks);
    for (uint32_t i = 0; i < blocks; i++) {
        remap[order[i]] = i;
        sorted[i] = representative[order[i]];
    }
    representative = std::move(sorted);
    for (uint32_t& block : block_of_triangle) {
        block = remap[block];
    }
    return blocks;
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
            if (it->second.baked_triangles < it->second.block_count) {
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
    uint64_t budget = MAX_TEXEL_STEPS_PER_UPDATE;

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
            entry.texture_size = vk::Extent2D{extent.width, extent.height};
            const uint32_t level = choose_subdivision_level(entry.texture_size, max_level);
            std::vector<uint32_t> block_of_triangle;
            entry.block_count = build_blocks(mesh, block_of_triangle, entry.block_representative);
            entry.usage = vk::MicromapUsageEXT{entry.block_count, level, MERIAN_OMM_FORMAT_4_STATE};
            entry.geometry_usage = vk::MicromapUsageEXT{mesh.geometry.primitive_count, level,
                                                        MERIAN_OMM_FORMAT_4_STATE};
            entry.vertices = mesh.geometry.vertices;
            entry.indices = mesh.geometry.indices;
            entry.alpha_texture_id = alpha_texture_of(mesh);
            entry.primitive_count = mesh.geometry.primitive_count;
            entry.baked_triangles = 0;

            entry.index_buffer = allocator->create_buffer(
                cmd, block_of_triangle.size() * sizeof(uint32_t),
                vk::BufferUsageFlagBits::eShaderDeviceAddress |
                    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                block_of_triangle.data(), MemoryMappingType::NONE, "micromap indices",
                INPUT_ALIGNMENT);

            entry.data = allocator->create_buffer(
                vk::DeviceSize(entry.block_count) * words_per_triangle(level) * sizeof(uint32_t),
                vk::BufferUsageFlagBits::eStorageBuffer |
                    vk::BufferUsageFlagBits::eShaderDeviceAddress |
                    vk::BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT,
                MemoryMappingType::NONE, "micromap data", INPUT_ALIGNMENT);
            // the bake only sets bits, so the states it leaves alone have to start transparent
            cmd->fill(entry.data, 0);

            // where the build finds each triangle's states; a fixed stride, so the host knows it
            std::vector<vk::MicromapTriangleEXT> array(entry.block_count);
            for (uint32_t i = 0; i < entry.block_count; i++) {
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
        const uint64_t per_triangle = bake_cost(level, entry.texture_size);
        const uint32_t first = entry.baked_triangles;
        // at least one triangle, so a mesh whose triangles each exceed the budget still finishes
        const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(
            std::max<uint64_t>(1, budget / per_triangle), entry.block_count - first));

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
            record.primitive_id = entry.block_representative[first + i];
            record.subdivision_level = level;
            record.data_word_offset = (first + i) * words_per_triangle(level);
            records.emplace_back(record);
        }

        entry.baked_triangles = first + count;
        budget -= std::min<uint64_t>(budget, uint64_t(count) * per_triangle);
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

    // 3. classify the slice, one workgroup per triangle and one dispatch per mesh
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

    // 4. build the micromap of every mesh whose last triangle was just baked
    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "build");
        bool any = false;
        for (const Slice& slice : slices) {
            Entry& entry = entries.at(slice.mesh_id);
            if (entry.baked_triangles < entry.block_count) {
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
    return it == entries.end() ? none : it->second.geometry_usage;
}

const BufferHandle& OpacityMicromaps::get_index_buffer(const uint32_t mesh_id) const {
    static const BufferHandle none;
    const auto it = entries.find(mesh_id);
    return it == entries.end() ? none : it->second.index_buffer;
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
