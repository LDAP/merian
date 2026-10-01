#include "merian-shaders/scene/micromaps.hpp"

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
#include <limits>
#include <numeric>
#include <unordered_map>

namespace merian {

namespace {
constexpr uint32_t STATES_PER_WORD = 16;
constexpr vk::DeviceSize INPUT_ALIGNMENT = MicromapBuilder::INPUT_ALIGNMENT;

constexpr vk::DeviceSize MAX_BYTES_PER_TRIANGLE = 64;

constexpr uint64_t MAX_TEXEL_STEPS_PER_UPDATE = 1'600'000'000;

constexpr vk::DeviceSize MIN_RECLAIMED_WORDS = 32ull << 20;

constexpr double EMISSION_TARGET_EDGE_TEXELS = 1.0;

uint32_t micro_triangles(const uint32_t subdivision_level) {
    return 1u << (2 * subdivision_level);
}

uint32_t words_per_block(const uint32_t subdivision_level) {
    return std::max(1u, micro_triangles(subdivision_level) / STATES_PER_WORD);
}

uint32_t supported_opacity_subdivision_level(const ContextHandle& context) {
    if (!Micromaps::is_opacity_supported(context)) {
        return 0;
    }
    uint32_t level = context->get_physical_device()
                         ->get_properties()
                         .get_opacity_micromap_properties_ext()
                         .maxOpacity4StateSubdivisionLevel;
    while (level > 0 && words_per_block(level) * sizeof(uint32_t) > MAX_BYTES_PER_TRIANGLE) {
        level--;
    }
    return level;
}

uint32_t choose_subdivision_level(const vk::Extent2D texture_size,
                                  const double target_edge_texels,
                                  const uint32_t max_level) {
    const double edge = std::max(texture_size.width, texture_size.height) / target_edge_texels;
    uint32_t level = 0;
    while (level < max_level && std::exp2(level) < edge) {
        level++;
    }
    return level;
}

// The blit that builds a mip skips texels unless it halves the size exactly.
uint32_t covering_max_mip(const ImageHandle& image) {
    const vk::Extent3D extent = image->get_extent();
    const auto halves_exactly = [](const uint32_t size) { return size == 1 || size % 2 == 0; };
    uint32_t mip = 0;
    while (mip + 1 < image->get_mip_levels() && halves_exactly(std::max(1u, extent.width >> mip)) &&
           halves_exactly(std::max(1u, extent.height >> mip))) {
        mip++;
    }
    return mip;
}

double footprint_texels(const double extent) {
    const double bounded = std::min(extent + 3, static_cast<double>(MERIAN_MICROMAP_MAX_SPAN));
    return bounded * bounded;
}

double micro_triangle_scan_cost(const uint32_t level, const double extent) {
    return micro_triangles(level) * footprint_texels(extent / std::exp2(level));
}

double scan_cost(const uint32_t level, const double extent) {
    return footprint_texels(extent) + micro_triangle_scan_cost(level, extent);
}

vk::Extent2D texture_size(const TextureManager& textures, const TextureID texture) {
    const vk::Extent3D extent = textures.get_texture(texture)->get_image()->get_extent();
    return {extent.width, extent.height};
}

bool is_magnified_nearest(const TextureManager& textures, const TextureID texture) {
    const SamplerHandle& sampler = textures.get_texture(texture)->get_sampler();
    return sampler && sampler->get_create_info().magFilter == vk::Filter::eNearest;
}

class HostGeometry {
  public:
    explicit HostGeometry(const Micromaps::MeshGeometry& mesh) : mesh(mesh) {}

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
        return float2(static_cast<float>(uv.x), static_cast<float>(uv.y));
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

    const Micromaps::MeshGeometry& mesh;
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
std::vector<uint32_t> build_blocks(const Micromaps::MeshGeometry& mesh,
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
        order[block] = (static_cast<uint64_t>(code) << 32) | block;
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

Micromaps::Micromaps(const ShaderCompileContextHandle& compile_context,
                     const ContextHandle& context,
                     const ResourceAllocatorHandle& allocator)
    : compile_context(compile_context), context(context), allocator(allocator),
      max_opacity_subdivision_level(supported_opacity_subdivision_level(context)),
      max_blocks_per_update(context->get_physical_device()
                                ->get_properties()
                                .get_properties()
                                .limits.maxComputeWorkGroupCount[0]),
      obj_allocator(std::make_shared<SimpleShaderObjectAllocator>(allocator)) {
    if (is_opacity_supported(context)) {
        builder.emplace(context, allocator);
    }
    opacity_pool.usage = vk::BufferUsageFlagBits::eShaderDeviceAddress |
                         vk::BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT |
                         vk::BufferUsageFlagBits::eTransferSrc;
    opacity_pool.name = "opacity micromap data";
    emission_pool.usage =
        vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferSrc;
    emission_pool.name = "emission micromap data";
}

size_t Micromaps::BlockKeyHash::operator()(const BlockKey& key) const noexcept {
    return hash_val(key.alpha_texture_serial, key.emission_texture_serial, key.material_content,
                    key.level, key.opacity, key.emission, key.uv[0], key.uv[1], key.uv[2]);
}

bool Micromaps::is_opacity_supported(const ContextHandle& context) {
    return MicromapBuilder::is_supported(context);
}

void Micromaps::ensure_pipeline(const SlangCompositionHandle& material_system_composition) {
    if (composition) {
        return;
    }
    composition = SlangComposition::create();
    composition->add_composition(material_system_composition);
    composition->add_module_from_path("merian-shaders/scene/micromap-bake.slang", true);
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

bool Micromaps::is_current(const Entry& entry,
                           const MeshGeometry& mesh,
                           const MaterialSystemHandle& material_system) {
    if (entry.mesh_serial != mesh.mesh_serial || entry.vertices != mesh.geometry.vertices ||
        entry.indices != mesh.geometry.indices ||
        entry.primitive_count != mesh.geometry.primitive_count ||
        entry.uv_version != mesh.uv_version || entry.alpha_tested != mesh.alpha_tested ||
        entry.opacity != mesh.opacity || entry.emission != mesh.emission) {
        return false;
    }
    const MaterialID material_id = mesh.geometry.material_id;
    const TextureManager& textures = *material_system->get_texture_manager();
    if (entry.alpha_tested) {
        const TextureID alpha_texture_id = material_system->get_alpha_texture_id(material_id);
        if (entry.alpha_texture_id != alpha_texture_id ||
            entry.alpha_texture_serial != textures.get_texture_serial(alpha_texture_id)) {
            return false;
        }
    }
    if (entry.emission) {
        const TextureID emission_texture_id = material_system->get_emission_texture_id(material_id);
        if (entry.emission_texture_id != emission_texture_id ||
            entry.emission_texture_serial != textures.get_texture_serial(emission_texture_id) ||
            !std::ranges::equal(entry.material_data,
                                material_system->get_material_data(material_id))) {
            return false;
        }
    }
    return true;
}

uint32_t Micromaps::allocate(Pool& pool, const uint32_t words) {
    const uint32_t offset = static_cast<uint32_t>(pool.used_words);
    pool.used_words += words;
    return offset;
}

void Micromaps::ensure_capacity(const CommandBufferHandle& cmd, Pool& pool) {
    const vk::DeviceSize needed = pool.used_words * sizeof(uint32_t);
    if (needed == 0 || (pool.buffer && pool.buffer->get_size() >= needed)) {
        return;
    }
    const vk::DeviceSize size = std::max(needed, pool.buffer ? 2 * pool.buffer->get_size() : 0);
    const BufferHandle grown = allocator->create_buffer(size, pool.usage, MemoryMappingType::NONE,
                                                        pool.name, INPUT_ALIGNMENT);
    if (pool.buffer) {
        cmd->copy(pool.buffer, grown);
        cmd->keep_until_pool_reset(pool.buffer);
    }
    pool.buffer = grown;
}

uint32_t Micromaps::intern_material_content(const std::span<const uint8_t> material_data) {
    const auto [it, inserted] =
        material_contents.try_emplace(std::string(material_data.begin(), material_data.end()),
                                      static_cast<uint32_t>(material_contents.size()));
    return it->second;
}

Micromaps::Entry Micromaps::create_entry(const CommandBufferHandle& cmd,
                                         const MeshGeometry& mesh,
                                         const MaterialSystemHandle& material_system) {
    const MaterialID material_id = mesh.geometry.material_id;
    const TextureManager& textures = *material_system->get_texture_manager();
    Entry entry{};
    entry.mesh_serial = mesh.mesh_serial;
    entry.vertices = mesh.geometry.vertices;
    entry.indices = mesh.geometry.indices;
    entry.primitive_count = mesh.geometry.primitive_count;
    entry.uv_version = mesh.uv_version;
    entry.alpha_tested = mesh.alpha_tested;
    entry.opacity = mesh.opacity;
    entry.emission = mesh.emission;

    entry.alpha_texture_id = TextureID(-1);
    uint32_t bake_flags = 0;
    if (entry.alpha_tested) {
        entry.alpha_texture_id = material_system->get_alpha_texture_id(material_id);
        entry.alpha_texture_serial = textures.get_texture_serial(entry.alpha_texture_id);
        entry.alpha_size = texture_size(textures, entry.alpha_texture_id);
        if (is_magnified_nearest(textures, entry.alpha_texture_id)) {
            bake_flags |= MicromapBakeFlags::AlphaNearestFilter;
        }
    }

    uint32_t level = 0;
    uint32_t max_level = std::numeric_limits<uint32_t>::max();
    uint32_t material_content = NO_OFFSET;
    if (entry.opacity) {
        level = choose_subdivision_level(entry.alpha_size, MERIAN_OMM_TARGET_EDGE_TEXELS,
                                         max_opacity_subdivision_level);
        max_level = max_opacity_subdivision_level;
    }
    if (entry.emission) {
        entry.emission_texture_id = material_system->get_emission_texture_id(material_id);
        entry.emission_texture_serial = textures.get_texture_serial(entry.emission_texture_id);
        entry.emission_size = texture_size(textures, entry.emission_texture_id);
        entry.emission_max_mip =
            covering_max_mip(textures.get_texture(entry.emission_texture_id)->get_image());
        if (is_magnified_nearest(textures, entry.emission_texture_id)) {
            bake_flags |= MicromapBakeFlags::EmissionNearestFilter;
        }
        const std::span<const uint8_t> material_data =
            material_system->get_material_data(material_id);
        entry.material_data.assign(material_data.begin(), material_data.end());
        material_content = intern_material_content(material_data);
        level = std::max(level,
                         choose_subdivision_level(entry.emission_size, EMISSION_TARGET_EDGE_TEXELS,
                                                  EMISSION_MAX_LEVEL));
        max_level = std::min(max_level, EMISSION_MAX_LEVEL);
    }
    entry.bake_flags = static_cast<MicromapBakeFlags>(bake_flags);
    entry.subdivision_level = std::min(level, max_level);
    level = entry.subdivision_level;
    entry.geometry_usage =
        vk::MicromapUsageEXT{mesh.geometry.primitive_count, level, MERIAN_OMM_FORMAT_4_STATE};

    std::vector<uint32_t> block_of_triangle;
    entry.block_triangle = build_blocks(mesh, block_of_triangle);

    const HostGeometry host(mesh);
    const auto texel_extent = [&](const uint32_t triangle, const vk::Extent2D size) {
        if (!host.is_available()) {
            return static_cast<double>(std::max(size.width, size.height));
        }
        const float2 uv0 = host.uv(triangle, 0);
        const float2 uv1 = host.uv(triangle, 1);
        const float2 uv2 = host.uv(triangle, 2);
        const float2 uv_extent = max(uv0, max(uv1, uv2)) - min(uv0, min(uv1, uv2));
        return std::max(static_cast<double>(uv_extent.x) * size.width,
                        static_cast<double>(uv_extent.y) * size.height);
    };
    const auto add_block = [&](const uint32_t triangle) {
        double cost = 0.;
        const double alpha_extent =
            entry.alpha_tested ? texel_extent(triangle, entry.alpha_size) : 0.;
        if (entry.opacity) {
            cost += scan_cost(level, alpha_extent);
        }
        if (entry.emission) {
            const double extent = texel_extent(triangle, entry.emission_size);
            const uint32_t mip =
                emission_scan_mip(static_cast<float>(extent), level, entry.emission_max_mip);
            cost += scan_cost(level, extent / std::exp2(mip));
            if (entry.alpha_tested) {
                cost += micro_triangle_scan_cost(level, alpha_extent);
            }
        }
        const uint32_t opacity_words = entry.opacity ? words_per_block(level) : 0;
        const uint32_t emission_words = entry.emission ? emission_record_words(level) : 0;
        blocks.emplace_back(Block{
            entry.opacity ? allocate(opacity_pool, opacity_words) : NO_OFFSET,
            entry.emission ? allocate(emission_pool, emission_words) : NO_OFFSET,
            opacity_words + emission_words,
            static_cast<uint32_t>(std::min(cost, static_cast<double>(UINT32_MAX))), 0, false});
        return static_cast<uint32_t>(blocks.size() - 1);
    };
    entry.blocks.resize(entry.block_triangle.size());
    for (uint32_t local = 0; local < entry.blocks.size(); local++) {
        const uint32_t triangle = entry.block_triangle[local];
        uint32_t block;
        if (host.is_available()) {
            const auto [it, inserted] = block_of_key.try_emplace(
                BlockKey{entry.alpha_texture_serial,
                         entry.emission_texture_serial,
                         material_content,
                         level,
                         entry.opacity,
                         entry.emission,
                         {host.uv_bits(triangle, 0), host.uv_bits(triangle, 1),
                          host.uv_bits(triangle, 2)}},
                0);
            if (inserted) {
                it->second = add_block(triangle);
            }
            block = it->second;
        } else {
            block = add_block(triangle);
        }
        entry.blocks[local] = block;
        if (blocks[block].references++ == 0) {
            live_words += blocks[block].words;
        }
    }

    if (entry.opacity) {
        entry.index_buffer = allocator->create_buffer(
            cmd, block_of_triangle.size() * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
                vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
            block_of_triangle.data(), MemoryMappingType::NONE, "micromap indices", INPUT_ALIGNMENT);
    }
    if (entry.emission) {
        std::vector<uint32_t> offsets(block_of_triangle.size());
        for (uint32_t triangle = 0; triangle < offsets.size(); triangle++) {
            offsets[triangle] = blocks[entry.blocks[block_of_triangle[triangle]]].emission_offset;
        }
        entry.emission_offsets =
            allocator->create_buffer(cmd, offsets, vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                     "emission micromap offsets");
    }
    return entry;
}

void Micromaps::release(const CommandBufferHandle& cmd, const Entry& entry) {
    for (const uint32_t block : entry.blocks) {
        if (--blocks[block].references == 0) {
            live_words -= blocks[block].words;
        }
    }
    cmd->keep_until_pool_reset(entry.micromap);
    cmd->keep_until_pool_reset(entry.index_buffer);
    cmd->keep_until_pool_reset(entry.emission_offsets);
}

void Micromaps::clear(const CommandBufferHandle& cmd, std::vector<uint32_t>& changed) {
    for (const auto& [mesh_id, entry] : entries) {
        if (entry.micromap) {
            changed.push_back(mesh_id);
        }
        release(cmd, entry);
    }
    entries.clear();
    block_of_key.clear();
    material_contents.clear();
    blocks.clear();
    live_words = 0;
    for (Pool* pool : {&opacity_pool, &emission_pool}) {
        cmd->keep_until_pool_reset(std::move(pool->buffer));
        pool->used_words = 0;
    }
}

void Micromaps::update(const CommandBufferHandle& cmd,
                       const std::vector<MeshGeometry>& meshes,
                       const MaterialSystemHandle& material_system,
                       std::vector<uint32_t>& changed) {
    const float threshold = material_system->get_alpha_test_threshold();
    if (threshold != alpha_threshold) {
        clear(cmd, changed);
        alpha_threshold = threshold;
    }

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
        if (it->second.micromap) {
            changed.push_back(it->first);
        }
        release(cmd, it->second);
        it = entries.erase(it);
    }

    // 1. new meshes
    const vk::DeviceSize opacity_used = opacity_pool.used_words;
    const vk::DeviceSize emission_used = emission_pool.used_words;
    const auto create_entries = [&] {
        for (const MeshGeometry& mesh : meshes) {
            if (!entries.contains(mesh.mesh_id)) {
                entries.try_emplace(mesh.mesh_id, create_entry(cmd, mesh, material_system));
            }
        }
    };
    create_entries();
    const vk::DeviceSize unused_words =
        opacity_pool.used_words + emission_pool.used_words - live_words;
    if (unused_words > std::max(live_words, MIN_RECLAIMED_WORDS)) {
        clear(cmd, changed);
        create_entries();
    }
    ensure_capacity(cmd, opacity_pool);
    ensure_capacity(cmd, emission_pool);
    const bool grown =
        opacity_pool.used_words != opacity_used || emission_pool.used_words != emission_used;

    // 2. bake
    std::vector<MicromapBakeJob> jobs;
    std::vector<MicromapBakeBlock> bake_blocks;
    std::vector<uint32_t> ready;
    uint64_t budget = MAX_TEXEL_STEPS_PER_UPDATE;
    for (const MeshGeometry& mesh : meshes) {
        Entry& entry = entries.at(mesh.mesh_id);
        if (entry.ready) {
            continue;
        }
        uint32_t job_index = NO_OFFSET;
        bool complete = true;
        for (uint32_t local = 0; local < entry.blocks.size(); local++) {
            Block& block = blocks[entry.blocks[local]];
            if (block.baked) {
                continue;
            }
            if (budget == 0 || bake_blocks.size() == max_blocks_per_update) {
                complete = false;
                break;
            }
            if (job_index == NO_OFFSET) {
                job_index = static_cast<uint32_t>(jobs.size());
                jobs.emplace_back(MicromapBakeJob{
                    mesh.geometry,
                    entry.opacity ? opacity_pool.buffer->get_device_address()
                                  : vk::DeviceAddress{0},
                    entry.emission ? emission_pool.buffer->get_device_address()
                                   : vk::DeviceAddress{0},
                    entry.subdivision_level, entry.alpha_size.width, entry.alpha_size.height,
                    entry.emission_size.width, entry.emission_size.height, entry.emission_max_mip,
                    entry.bake_flags, 0});
            }
            bake_blocks.emplace_back(MicromapBakeBlock{job_index, entry.block_triangle[local],
                                                       block.opacity_offset,
                                                       block.emission_offset});
            block.baked = true;
            budget -= std::min<uint64_t>(budget, block.cost);
        }
        if (complete) {
            entry.ready = true;
            entry.block_triangle = {};
            ready.push_back(mesh.mesh_id);
        }
    }
    if (bake_blocks.empty() && ready.empty() && !grown) {
        return;
    }

    MERIAN_PROFILE_SCOPE_GPU(cmd, "Micromaps::update");
    const vk::AccessFlags2 micromap_read =
        builder ? vk::AccessFlagBits2::eMicromapReadEXT : vk::AccessFlags2{};
    // the uploads and a grown pool's copy
    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
            vk::AccessFlagBits2::eTransferRead | micromap_read});

    if (!bake_blocks.empty()) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "bake");
        ensure_pipeline(material_system->get_composition());
        const BufferHandle job_buffer = allocator->create_buffer(
            cmd, jobs.size() * sizeof(MicromapBakeJob),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            jobs.data(), MemoryMappingType::NONE, "micromap bake jobs");
        const BufferHandle block_buffer = allocator->create_buffer(
            cmd, bake_blocks.size() * sizeof(MicromapBakeBlock),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            bake_blocks.data(), MemoryMappingType::NONE, "micromap bake blocks");
        cmd->keep_until_pool_reset(job_buffer);
        cmd->keep_until_pool_reset(block_buffer);
        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead});

        const auto ep = bake_entry_point.get();
        const auto pipe = bake_pipeline.get();
        const auto& params = bake_params.get();
        auto c = params->get_cursor();
        c["jobs"] = job_buffer;
        c["blocks"] = block_buffer;
        c["block_count"] = static_cast<uint32_t>(bake_blocks.size());

        cmd->bind(pipe);
        ep->bind("material_system", material_system->get_shader_object(), cmd, pipe, obj_allocator);
        ep->bind("params", params, cmd, pipe, obj_allocator);
        cmd->dispatch(static_cast<uint32_t>(bake_blocks.size()), 1, 1);

        // the baked records and states
        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead | micromap_read});
    }

    // 3. opacity micromaps
    bool build = false;
    for (const uint32_t mesh_id : ready) {
        Entry& entry = entries.at(mesh_id);
        if (!entry.opacity || !builder) {
            continue;
        }
        const uint32_t level = entry.subdivision_level;
        std::vector<vk::MicromapTriangleEXT> triangle_array(entry.blocks.size());
        for (uint32_t local = 0; local < entry.blocks.size(); local++) {
            triangle_array[local].dataOffset = blocks[entry.blocks[local]].opacity_offset *
                                               static_cast<uint32_t>(sizeof(uint32_t));
            triangle_array[local].subdivisionLevel = static_cast<uint16_t>(level);
            triangle_array[local].format = MERIAN_OMM_FORMAT_4_STATE;
        }
        const BufferHandle triangles = allocator->create_buffer(
            cmd, triangle_array.size() * sizeof(vk::MicromapTriangleEXT),
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
                vk::BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT,
            triangle_array.data(), MemoryMappingType::NONE, "micromap triangles", INPUT_ALIGNMENT);
        const vk::MicromapUsageEXT usage{static_cast<uint32_t>(entry.blocks.size()), level,
                                         MERIAN_OMM_FORMAT_4_STATE};
        entry.micromap =
            builder->queue_build(usage, opacity_pool.buffer, 0, triangles, 0, "opacity micromap");
        cmd->keep_until_pool_reset(triangles);
        cmd->keep_until_pool_reset(opacity_pool.buffer);
        changed.push_back(mesh_id);
        build = true;
    }
    if (!build) {
        return;
    }

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eMicromapBuildEXT, vk::AccessFlagBits2::eMicromapReadEXT});
    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "build");
        builder->get_cmds(cmd);
    }

    cmd->barrier(vk::MemoryBarrier2{vk::PipelineStageFlagBits2::eMicromapBuildEXT,
                                    vk::AccessFlagBits2::eMicromapWriteEXT,
                                    vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                    vk::AccessFlagBits2::eMicromapReadEXT});
}

bool Micromaps::get_opacity(const uint32_t mesh_id,
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

std::optional<Micromaps::EmissionMicromap> Micromaps::get_emission(const uint32_t mesh_id) const {
    const auto it = entries.find(mesh_id);
    if (it == entries.end() || !it->second.ready || !it->second.emission_offsets) {
        return std::nullopt;
    }
    const Entry& entry = it->second;
    return EmissionMicromap{entry.emission_offsets->get_device_address(),
                            emission_pool.buffer->get_device_address(), entry.subdivision_level};
}

void Micromaps::properties(Properties& props) {
    uint32_t opacity = 0;
    uint32_t emission = 0;
    uint32_t baking = 0;
    for (const auto& [mesh_id, entry] : entries) {
        if (!entry.ready) {
            baking++;
            continue;
        }
        opacity += entry.micromap ? 1 : 0;
        emission += entry.emission_offsets ? 1 : 0;
    }
    const vk::DeviceSize size = (opacity_pool.buffer ? opacity_pool.buffer->get_size() : 0) +
                                (emission_pool.buffer ? emission_pool.buffer->get_size() : 0);
    props.output_text(fmt::format(
        "opacity micromaps: {}\nemission micromaps: {}\nbaking: {}\nshared blocks: {}\nsize: {}",
        opacity, emission, baking, blocks.size(), format_size(size)));
}

} // namespace merian
