#include "merian-shaders/scene/opacity_micromaps.hpp"

#include "merian/utils/hash.hpp"
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
constexpr vk::DeviceSize INPUT_ALIGNMENT = MicromapBuilder::INPUT_ALIGNMENT;

constexpr vk::DeviceSize MAX_BYTES_PER_TRIANGLE = 64;

constexpr uint64_t MAX_TEXEL_STEPS_PER_UPDATE = 1'600'000'000;

uint32_t micro_triangles(const uint32_t subdivision_level) {
    return 1u << (2 * subdivision_level);
}

uint32_t words_per_block(const uint32_t subdivision_level) {
    return std::max(1u, micro_triangles(subdivision_level) / STATES_PER_WORD);
}

uint32_t choose_subdivision_level(const vk::Extent2D texture_size, const uint32_t max_level) {
    const double edge =
        std::max(texture_size.width, texture_size.height) / double(MERIAN_OMM_TARGET_EDGE_TEXELS);
    uint32_t level = 0;
    while (level < max_level && std::exp2(level) < edge &&
           words_per_block(level + 1) * sizeof(uint32_t) <= MAX_BYTES_PER_TRIANGLE) {
        level++;
    }
    return level;
}

uint32_t bake_cost(const uint32_t level, const double extent) {
    const auto scan = [](const double span) {
        const double bounded = std::min(span + 3, double(MERIAN_OMM_MAX_SPAN));
        return bounded * bounded;
    };
    const double cost = scan(extent) + micro_triangles(level) * scan(extent / std::exp2(level));
    return static_cast<uint32_t>(std::min(cost, double(UINT32_MAX)));
}

class HostGeometry {
  public:
    explicit HostGeometry(const OpacityMicromaps::MeshGeometry& mesh) : mesh(mesh) {}

    bool is_available() const {
        return mesh.host_vertices != nullptr;
    }

    uint32_t uv_bits(const uint32_t triangle, const uint32_t corner) const {
        uint32_t bits;
        std::memcpy(&bits, &mesh.host_vertices[vertex(3 * triangle + corner)].uv, sizeof(bits));
        return bits;
    }

    float2 uv(const uint32_t triangle, const uint32_t corner) const {
        const half2& uv = mesh.host_vertices[vertex(3 * triangle + corner)].uv;
        return float2(float(uv.x), float(uv.y));
    }

  private:
    uint32_t vertex(const uint32_t index) const {
        switch (mesh.host_index_type) {
        case vk::IndexType::eUint32:
            return static_cast<const uint32_t*>(mesh.host_indices)[index];
        case vk::IndexType::eUint16:
            return static_cast<const uint16_t*>(mesh.host_indices)[index];
        case vk::IndexType::eUint8:
            return static_cast<const uint8_t*>(mesh.host_indices)[index];
        default:
            return index;
        }
    }

    const OpacityMicromaps::MeshGeometry& mesh;
};

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

// Reuse pre-pass and spatial sort of NVIDIA's Opacity Micro-Map SDK.
std::vector<uint32_t> build_blocks(const OpacityMicromaps::MeshGeometry& mesh,
                                   std::vector<uint32_t>& block_of_triangle) {
    const uint32_t count = mesh.geometry.primitive_count;
    block_of_triangle.resize(count);

    const HostGeometry host(mesh);
    if (!host.is_available()) {
        std::iota(block_of_triangle.begin(), block_of_triangle.end(), 0u);
        std::vector<uint32_t> block_triangle(count);
        std::iota(block_triangle.begin(), block_triangle.end(), 0u);
        return block_triangle;
    }

    using Key = std::array<uint32_t, 3>;
    struct KeyHash {
        size_t operator()(const Key& key) const noexcept {
            return hash_val(key[0], key[1], key[2]);
        }
    };
    std::unordered_map<Key, uint32_t, KeyHash> block_of_key;
    std::vector<uint32_t> block_triangle;
    for (uint32_t triangle = 0; triangle < count; triangle++) {
        const Key key{host.uv_bits(triangle, 0), host.uv_bits(triangle, 1),
                      host.uv_bits(triangle, 2)};
        const auto [it, inserted] =
            block_of_key.try_emplace(key, static_cast<uint32_t>(block_triangle.size()));
        block_of_triangle[triangle] = it->second;
        if (inserted) {
            block_triangle.push_back(triangle);
        }
    }

    const uint32_t block_count = static_cast<uint32_t>(block_triangle.size());
    std::vector<uint64_t> order(block_count);
    for (uint32_t block = 0; block < block_count; block++) {
        const uint32_t triangle = block_triangle[block];
        const float2 centroid =
            (host.uv(triangle, 0) + host.uv(triangle, 1) + host.uv(triangle, 2)) / 3.f;
        const float2 wrapped = centroid - floor(centroid);
        const uint32_t code = std::isfinite(wrapped.x) && std::isfinite(wrapped.y)
                                  ? morton(static_cast<uint32_t>(wrapped.x * 65535.f),
                                           static_cast<uint32_t>(wrapped.y * 65535.f))
                                  : 0;
        order[block] = (uint64_t(code) << 32) | block;
    }
    std::sort(order.begin(), order.end());

    std::vector<uint32_t> sorted_block(block_count);
    std::vector<uint32_t> sorted_triangle(block_count);
    for (uint32_t i = 0; i < block_count; i++) {
        const uint32_t block = static_cast<uint32_t>(order[i]);
        sorted_block[block] = i;
        sorted_triangle[i] = block_triangle[block];
    }
    for (uint32_t& block : block_of_triangle) {
        block = sorted_block[block];
    }
    return sorted_triangle;
}

} // namespace

OpacityMicromaps::OpacityMicromaps(const ShaderCompileContextHandle& compile_context,
                                   const ContextHandle& context,
                                   const ResourceAllocatorHandle& allocator)
    : compile_context(compile_context), context(context), allocator(allocator),
      max_subdivision_level(context->get_physical_device()
                                ->get_properties()
                                .get_opacity_micromap_properties_ext()
                                .maxOpacity4StateSubdivisionLevel),
      max_blocks_per_update(context->get_physical_device()
                                ->get_properties()
                                .get_properties()
                                .limits.maxComputeWorkGroupCount[0]),
      builder(context, allocator),
      obj_allocator(std::make_shared<SimpleShaderObjectAllocator>(allocator)) {}

bool OpacityMicromaps::is_supported(const ContextHandle& context) {
    return MicromapBuilder::is_supported(context);
}

void OpacityMicromaps::ensure_pipeline(const SlangCompositionHandle& material_system_composition) {
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
    bake_params = Versioned<ShaderObject>([this] {
        return bake_entry_point->create_shader_object_for_parameter(context, "params", allocator);
    });
    bake_params.depends_on(bake_entry_point);
}

bool OpacityMicromaps::is_current(const Entry& entry,
                                  const MeshGeometry& mesh,
                                  const MaterialSystemHandle& material_system) {
    return entry.vertices == mesh.geometry.vertices && entry.indices == mesh.geometry.indices &&
           entry.primitive_count == mesh.geometry.primitive_count &&
           entry.uv_version == mesh.uv_version &&
           entry.alpha_texture_id ==
               material_system->get_alpha_texture_id(mesh.geometry.material_id) &&
           entry.alpha_threshold == material_system->get_alpha_test_threshold();
}

OpacityMicromaps::Entry
OpacityMicromaps::create_entry(const CommandBufferHandle& cmd,
                               const MeshGeometry& mesh,
                               const MaterialSystemHandle& material_system) const {
    Entry entry{};
    entry.vertices = mesh.geometry.vertices;
    entry.indices = mesh.geometry.indices;
    entry.primitive_count = mesh.geometry.primitive_count;
    entry.uv_version = mesh.uv_version;
    entry.alpha_texture_id = material_system->get_alpha_texture_id(mesh.geometry.material_id);
    entry.alpha_threshold = material_system->get_alpha_test_threshold();

    const vk::Extent3D extent = material_system->get_texture_manager()
                                    ->get_texture(entry.alpha_texture_id)
                                    ->get_image()
                                    ->get_extent();
    const vk::Extent2D texture_size{extent.width, extent.height};
    entry.subdivision_level = choose_subdivision_level(texture_size, max_subdivision_level);
    const uint32_t level = entry.subdivision_level;

    std::vector<uint32_t> block_of_triangle;
    entry.block_triangle = build_blocks(mesh, block_of_triangle);
    entry.block_count = static_cast<uint32_t>(entry.block_triangle.size());
    entry.geometry_usage =
        vk::MicromapUsageEXT{mesh.geometry.primitive_count, level, MERIAN_OMM_FORMAT_4_STATE};

    const HostGeometry host(mesh);
    entry.block_cost.resize(entry.block_count);
    for (uint32_t block = 0; block < entry.block_count; block++) {
        double texel_extent = std::max(texture_size.width, texture_size.height);
        if (host.is_available()) {
            const uint32_t triangle = entry.block_triangle[block];
            const float2 uv0 = host.uv(triangle, 0);
            const float2 uv1 = host.uv(triangle, 1);
            const float2 uv2 = host.uv(triangle, 2);
            const float2 uv_extent = max(uv0, max(uv1, uv2)) - min(uv0, min(uv1, uv2));
            texel_extent = std::max(double(uv_extent.x) * texture_size.width,
                                    double(uv_extent.y) * texture_size.height);
        }
        entry.block_cost[block] = bake_cost(level, texel_extent);
    }

    entry.index_buffer = allocator->create_buffer(
        cmd, block_of_triangle.size() * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
        block_of_triangle.data(), MemoryMappingType::NONE, "micromap indices", INPUT_ALIGNMENT);

    entry.data = allocator->create_buffer(
        vk::DeviceSize(entry.block_count) * words_per_block(level) * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT,
        MemoryMappingType::NONE, "micromap data", INPUT_ALIGNMENT);

    std::vector<vk::MicromapTriangleEXT> triangle_array(entry.block_count);
    for (uint32_t block = 0; block < entry.block_count; block++) {
        triangle_array[block].dataOffset =
            block * words_per_block(level) * static_cast<uint32_t>(sizeof(uint32_t));
        triangle_array[block].subdivisionLevel = static_cast<uint16_t>(level);
        triangle_array[block].format = MERIAN_OMM_FORMAT_4_STATE;
    }
    entry.triangle_array = allocator->create_buffer(
        cmd, triangle_array.size() * sizeof(vk::MicromapTriangleEXT),
        vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT,
        triangle_array.data(), MemoryMappingType::NONE, "micromap triangles", INPUT_ALIGNMENT);

    entry.baked_blocks = 0;
    return entry;
}

void OpacityMicromaps::update(const CommandBufferHandle& cmd,
                              const std::vector<MeshGeometry>& meshes,
                              const MaterialSystemHandle& material_system,
                              std::vector<uint32_t>& changed) {
    std::unordered_map<uint32_t, const MeshGeometry*> mesh_of_id;
    mesh_of_id.reserve(meshes.size());
    for (const MeshGeometry& mesh : meshes) {
        mesh_of_id.try_emplace(mesh.mesh_id, &mesh);
    }
    for (auto it = entries.begin(); it != entries.end();) {
        const auto mesh = mesh_of_id.find(it->first);
        if (mesh != mesh_of_id.end() && is_current(it->second, *mesh->second, material_system)) {
            ++it;
            continue;
        }
        const Entry& entry = it->second;
        if (entry.micromap) {
            changed.push_back(it->first);
        }
        cmd->keep_until_pool_reset(entry.micromap);
        cmd->keep_until_pool_reset(entry.index_buffer);
        cmd->keep_until_pool_reset(entry.data);
        it = entries.erase(it);
    }

    std::vector<OmmBakeJob> jobs;
    std::vector<OmmBakeBlock> blocks;
    std::vector<uint32_t> finished;
    uint64_t budget = MAX_TEXEL_STEPS_PER_UPDATE;
    for (const MeshGeometry& mesh : meshes) {
        if (budget == 0 || blocks.size() == max_blocks_per_update) {
            break;
        }
        auto it = entries.find(mesh.mesh_id);
        if (it == entries.end()) {
            it = entries.try_emplace(mesh.mesh_id, create_entry(cmd, mesh, material_system)).first;
        }
        Entry& entry = it->second;
        if (entry.baked_blocks == entry.block_count) {
            continue;
        }

        const uint32_t job_index = static_cast<uint32_t>(jobs.size());
        jobs.emplace_back(OmmBakeJob{mesh.geometry, entry.data->get_device_address(),
                                     entry.subdivision_level, 0});
        cmd->keep_until_pool_reset(entry.data);
        do {
            const uint32_t block = entry.baked_blocks++;
            blocks.emplace_back(OmmBakeBlock{job_index, entry.block_triangle[block],
                                             block * words_per_block(entry.subdivision_level)});
            budget -= std::min<uint64_t>(budget, entry.block_cost[block]);
        } while (budget > 0 && entry.baked_blocks < entry.block_count &&
                 blocks.size() < max_blocks_per_update);

        if (entry.baked_blocks == entry.block_count) {
            finished.push_back(mesh.mesh_id);
        }
    }
    if (blocks.empty()) {
        return;
    }

    MERIAN_PROFILE_SCOPE_GPU(cmd, "OpacityMicromaps::update");
    ensure_pipeline(material_system->get_composition());

    const BufferHandle job_buffer = allocator->create_buffer(
        cmd, jobs.size() * sizeof(OmmBakeJob),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        jobs.data(), MemoryMappingType::NONE, "micromap bake jobs");
    const BufferHandle block_buffer = allocator->create_buffer(
        cmd, blocks.size() * sizeof(OmmBakeBlock),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        blocks.data(), MemoryMappingType::NONE, "micromap bake blocks");
    cmd->keep_until_pool_reset(job_buffer);
    cmd->keep_until_pool_reset(block_buffer);

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eMicromapBuildEXT |
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        vk::AccessFlagBits2::eShaderRead});

    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "bake");
        const auto ep = bake_entry_point.get();
        const auto pipe = bake_pipeline.get();
        const auto& params = bake_params.get();
        auto c = params->get_cursor();
        c["jobs"] = job_buffer;
        c["blocks"] = block_buffer;
        c["block_count"] = static_cast<uint32_t>(blocks.size());

        cmd->bind(pipe);
        ep->bind("material_system", material_system->get_shader_object(), cmd, pipe, obj_allocator);
        ep->bind("params", params, cmd, pipe, obj_allocator);
        cmd->dispatch(static_cast<uint32_t>(blocks.size()), 1, 1);
    }

    if (finished.empty()) {
        return;
    }

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eMicromapBuildEXT, vk::AccessFlagBits2::eShaderRead});

    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "build");
        for (const uint32_t mesh_id : finished) {
            Entry& entry = entries.at(mesh_id);
            const vk::MicromapUsageEXT usage{entry.block_count, entry.subdivision_level,
                                             MERIAN_OMM_FORMAT_4_STATE};
            entry.micromap = builder.queue_build(usage, entry.data, 0, entry.triangle_array, 0,
                                                 "opacity micromap");
            entry.data.reset();
            entry.triangle_array.reset();
            entry.block_triangle = {};
            entry.block_cost = {};
            changed.push_back(mesh_id);
        }
        builder.get_cmds(cmd);
    }

    cmd->barrier(vk::MemoryBarrier2{vk::PipelineStageFlagBits2::eMicromapBuildEXT,
                                    vk::AccessFlagBits2::eMicromapWriteEXT,
                                    vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                    vk::AccessFlagBits2::eMicromapReadEXT});
}

void OpacityMicromaps::clear(const CommandBufferHandle& cmd, std::vector<uint32_t>& changed) {
    for (const auto& [mesh_id, entry] : entries) {
        if (entry.micromap) {
            changed.push_back(mesh_id);
        }
        cmd->keep_until_pool_reset(entry.micromap);
        cmd->keep_until_pool_reset(entry.index_buffer);
        cmd->keep_until_pool_reset(entry.data);
    }
    entries.clear();
}

bool OpacityMicromaps::get(const uint32_t mesh_id,
                           vk::AccelerationStructureTrianglesOpacityMicromapEXT& omm) const {
    const auto it = entries.find(mesh_id);
    if (it == entries.end() || !it->second.micromap) {
        return false;
    }
    const Entry& entry = it->second;
    omm.indexType = vk::IndexType::eUint32;
    omm.indexBuffer = entry.index_buffer->get_device_address();
    omm.indexStride = sizeof(uint32_t);
    omm.baseTriangle = 0;
    omm.usageCountsCount = 1;
    omm.pUsageCounts = &entry.geometry_usage;
    omm.micromap = **entry.micromap;
    return true;
}

void OpacityMicromaps::properties(Properties& props) {
    uint32_t built = 0;
    vk::DeviceSize size = 0;
    for (const auto& [mesh_id, entry] : entries) {
        if (entry.micromap) {
            built++;
            size += entry.micromap->get_size() + entry.index_buffer->get_size();
        }
    }
    props.output_text(fmt::format("micromaps: {}\nbaking: {}\nsize: {}", built,
                                  entries.size() - built, format_size(size)));
}

} // namespace merian
