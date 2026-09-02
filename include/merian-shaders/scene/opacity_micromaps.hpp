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

/*
 * Opacity micromaps for alpha-tested geometry, baked on the device from the same alpha test that
 * traversal runs, so no texture has to be readable by the host.
 *
 * A micro-triangle the bake proves entirely inside or entirely outside the cutout is settled
 * during traversal; the rest keep invoking the alpha test. The proof runs the alpha test itself at
 * every texel the micro-triangle covers, so the image does not change.
 *
 * Micromaps are kept per mesh across updates and only the meshes that changed are rebuilt.
 */
class OpacityMicromaps {
  public:
    // What one mesh needs to get a micromap. `geometry` addresses its vertices and indices.
    struct MeshGeometry {
        uint32_t mesh_id;
        GeometryData geometry;
    };

    OpacityMicromaps(const ShaderCompileContextHandle& compile_context,
                     const ContextHandle& context,
                     const ResourceAllocatorHandle& allocator);

    // Whether the device can use micromaps at all.
    static bool is_supported(const ContextHandle& context);

    // The highest subdivision level the device supports for the format used here.
    uint32_t max_subdivision_level() const;

    // Brings the micromaps up to date with `meshes`: drops the ones that are gone, leaves the
    // valid ones alone, and bakes a bounded slice of what is left, so a scene that streams its
    // geometry in spreads the cost over frames instead of stalling on it. A mesh whose bake
    // finished this call is appended to `built`, and the caller has to rebuild its acceleration
    // structure.
    void update(const CommandBufferHandle& cmd,
                const std::vector<MeshGeometry>& meshes,
                const MaterialSystemHandle& material_system,
                const uint32_t max_subdivision_level,
                const ShaderObjectAllocatorHandle& obj_allocator,
                std::vector<uint32_t>& built);

    // The micromap of a mesh, or an empty handle if it has none.
    const MicromapHandle& get(const uint32_t mesh_id) const;

    // How many triangles a mesh's micromap covers, and at which level and format.
    const vk::MicromapUsageEXT& get_usage(const uint32_t mesh_id) const;

    // Maps triangle i of a geometry to micromap triangle i. Empty while nothing was built.
    const BufferHandle& get_index_buffer() const {
        return index_buffer;
    }

    void clear();

    void properties(Properties& props);

  private:
    void ensure_pipelines(const SlangCompositionHandle& material_system_composition);

    // Shader objects are bound once each per command buffer, so every dispatch needs its own.
    const ShaderObjectHandle& next_object(std::vector<Versioned<ShaderObject>>& pool,
                                          uint32_t& used,
                                          const Versioned<SlangProgramEntryPoint>& entry_point);

    const ShaderCompileContextHandle compile_context;
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;

    MicromapBuilder builder;

    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    Versioned<SlangProgramEntryPoint> bake_entry_point;
    Versioned<Pipeline> bake_pipeline;
    std::vector<Versioned<ShaderObject>> bake_params;
    ShaderObjectAllocatorHandle fallback_obj_allocator;

    // 0, 1, 2, ... : triangle i of a geometry uses micromap triangle i, shared by all of them
    BufferHandle index_buffer;

    struct Entry {
        // empty until every triangle has been baked
        MicromapHandle micromap;
        vk::MicromapUsageEXT usage;
        // what the micromap was built from; a change of any of it invalidates it
        vk::DeviceAddress vertices;
        vk::DeviceAddress indices;
        uint32_t alpha_texture_id;
        vk::Extent2D texture_size;

        uint32_t primitive_count;
        // how many of them carry a state already; the bake resumes here
        uint32_t baked_triangles;
        // the build inputs, alive until the micromap is built
        BufferHandle data;
        BufferHandle triangles;
    };
    std::unordered_map<uint32_t, Entry> entries;

    vk::DeviceSize data_size = 0;
    uint32_t pending_count = 0;
};

} // namespace merian
