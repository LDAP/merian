#pragma once

#include "merian-shaders/scene/micromap-bake.slangh"
#include "merian-shaders/shading/materials/material_system.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/versioned.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/raytrace/micromap_builder.hpp"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace merian {

class Properties;

class Micromaps {
  public:
    struct MeshGeometry {
        uint32_t mesh_id;
        uint32_t mesh_serial;
        GeometryData geometry;
        uint32_t uv_version;
        bool alpha_tested;
        bool opacity;
        bool emission;

        const PackedVertexData* host_vertices = nullptr;
        const void* host_indices = nullptr;
        vk::IndexType host_index_type = vk::IndexType::eNoneKHR;
    };

    struct EmissionMicromap {
        vk::DeviceAddress offsets;
        vk::DeviceAddress records;
        uint32_t level;
    };

    Micromaps(const ShaderCompileContextHandle& compile_context,
              const ContextHandle& context,
              const ResourceAllocatorHandle& allocator);

    static bool is_opacity_supported(const ContextHandle& context);

    // changed receives the meshes whose opacity micromap appeared or went away.
    void update(const CommandBufferHandle& cmd,
                const std::vector<MeshGeometry>& meshes,
                const MaterialSystemHandle& material_system,
                std::vector<uint32_t>& changed);

    bool get_opacity(const uint32_t mesh_id,
                     vk::AccelerationStructureTrianglesOpacityMicromapEXT& omm) const;

    std::optional<EmissionMicromap> get_emission(const uint32_t mesh_id) const;

    void properties(Properties& props);

  private:
    struct BlockKey {
        uint64_t alpha_texture_serial;
        uint64_t emission_texture_serial;
        uint32_t material_content;
        uint32_t level;
        bool opacity;
        bool emission;
        std::array<uint32_t, 3> uv;

        bool operator==(const BlockKey&) const = default;
    };

    struct BlockKeyHash {
        size_t operator()(const BlockKey& key) const noexcept;
    };

    struct Block {
        uint32_t opacity_offset;
        uint32_t emission_offset;
        uint32_t words;
        uint32_t cost;
        uint32_t references;
        bool baked;
    };

    struct Pool {
        BufferHandle buffer;
        vk::DeviceSize used_words = 0;
        vk::BufferUsageFlags usage;
        std::string name;
    };

    struct Entry {
        uint32_t mesh_serial;
        vk::DeviceAddress vertices;
        vk::DeviceAddress indices;
        uint32_t primitive_count;
        uint32_t uv_version;
        bool alpha_tested;
        bool opacity;
        bool emission;
        TextureID alpha_texture_id;
        TextureID emission_texture_id;
        uint64_t alpha_texture_serial;
        uint64_t emission_texture_serial;
        std::vector<uint8_t> material_data;

        uint32_t subdivision_level;
        vk::Extent2D alpha_size;
        vk::Extent2D emission_size;
        uint32_t emission_max_mip;
        MicromapBakeFlags bake_flags;

        std::vector<uint32_t> blocks;
        std::vector<uint32_t> block_triangle;
        bool ready;

        BufferHandle index_buffer;
        BufferHandle emission_offsets;
        MicromapHandle micromap;
        vk::MicromapUsageEXT geometry_usage;
    };

    static constexpr uint32_t NO_OFFSET = ~0u;

    static bool is_current(const Entry& entry,
                           const MeshGeometry& mesh,
                           const MaterialSystemHandle& material_system);

    Entry create_entry(const CommandBufferHandle& cmd,
                       const MeshGeometry& mesh,
                       const MaterialSystemHandle& material_system);

    uint32_t intern_material_content(const std::span<const uint8_t> material_data);

    void release(const CommandBufferHandle& cmd, const Entry& entry);

    uint32_t allocate(Pool& pool, const uint32_t words);

    void ensure_capacity(const CommandBufferHandle& cmd, Pool& pool);

    void clear(const CommandBufferHandle& cmd, std::vector<uint32_t>& changed);

    void ensure_pipeline(const SlangCompositionHandle& material_system_composition);

    const ShaderCompileContextHandle compile_context;
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;

    const uint32_t max_opacity_subdivision_level;
    const uint32_t max_blocks_per_update;

    std::optional<MicromapBuilder> builder;

    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    Versioned<SlangProgramEntryPoint> bake_entry_point;
    Versioned<Pipeline> bake_pipeline;
    Versioned<ShaderObject> bake_params;
    ShaderObjectAllocatorHandle obj_allocator;

    std::unordered_map<uint32_t, Entry> entries;

    std::unordered_map<BlockKey, uint32_t, BlockKeyHash> block_of_key;
    std::unordered_map<std::string, uint32_t> material_contents;
    std::vector<Block> blocks;
    Pool opacity_pool;
    Pool emission_pool;
    vk::DeviceSize live_words = 0;
    float alpha_threshold = -1.f;
};

} // namespace merian
