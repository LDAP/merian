#pragma once

#include "merian-shaders/scene/omm-bake.slangh"
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

#include <unordered_map>
#include <vector>

namespace merian {

class Properties;

class OpacityMicromaps {
  public:
    struct MeshGeometry {
        uint32_t mesh_id;
        GeometryData geometry;
        uint32_t uv_version;

        const PackedVertexData* host_vertices = nullptr;
        const void* host_indices = nullptr;
        vk::IndexType host_index_type = vk::IndexType::eNoneKHR;
    };

    OpacityMicromaps(const ShaderCompileContextHandle& compile_context,
                     const ContextHandle& context,
                     const ResourceAllocatorHandle& allocator);

    static bool is_supported(const ContextHandle& context);

    void update(const CommandBufferHandle& cmd,
                const std::vector<MeshGeometry>& meshes,
                const MaterialSystemHandle& material_system,
                std::vector<uint32_t>& changed);

    void clear(const CommandBufferHandle& cmd, std::vector<uint32_t>& changed);

    bool get(const uint32_t mesh_id,
             vk::AccelerationStructureTrianglesOpacityMicromapEXT& omm) const;

    void properties(Properties& props);

  private:
    struct Entry {
        vk::DeviceAddress vertices;
        vk::DeviceAddress indices;
        uint32_t primitive_count;
        uint32_t uv_version;
        TextureID alpha_texture_id;
        float alpha_threshold;

        MicromapHandle micromap;
        vk::MicromapUsageEXT geometry_usage;
        BufferHandle index_buffer;

        uint32_t subdivision_level;
        uint32_t block_count;
        uint32_t baked_blocks;

        std::vector<uint32_t> block_triangle;
        std::vector<uint32_t> block_cost;
        BufferHandle data;
        BufferHandle triangle_array;
    };

    static bool is_current(const Entry& entry,
                           const MeshGeometry& mesh,
                           const MaterialSystemHandle& material_system);

    Entry create_entry(const CommandBufferHandle& cmd,
                       const MeshGeometry& mesh,
                       const MaterialSystemHandle& material_system) const;

    void ensure_pipeline(const SlangCompositionHandle& material_system_composition);

    const ShaderCompileContextHandle compile_context;
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;

    const uint32_t max_subdivision_level;
    const uint32_t max_blocks_per_update;

    MicromapBuilder builder;

    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    Versioned<SlangProgramEntryPoint> bake_entry_point;
    Versioned<Pipeline> bake_pipeline;
    Versioned<ShaderObject> bake_params;
    ShaderObjectAllocatorHandle obj_allocator;

    std::unordered_map<uint32_t, Entry> entries;
};

} // namespace merian
