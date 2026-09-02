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
 * A micro-triangle the bake finds entirely inside or entirely outside the cutout is settled during
 * traversal; the rest still invoke the alpha test. Where the bake is unsure the micro-triangle
 * stays unknown, so the image never depends on how finely it sampled.
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

    // The device features a node must ask for so that micromaps can be used.
    static std::vector<std::string> optional_device_features();

    // The highest subdivision level the device supports for the format used here.
    uint32_t max_subdivision_level() const;

    // Bakes and builds a micromap for every mesh in `meshes`, replacing all earlier ones. The
    // caller must barrier between this and the acceleration structure build that reads them.
    void build(const CommandBufferHandle& cmd,
               const std::vector<MeshGeometry>& meshes,
               const MaterialSystemHandle& material_system,
               const uint32_t subdivision_level,
               const uint32_t max_samples_per_edge,
               const ShaderObjectAllocatorHandle& obj_allocator);

    // The micromap of a mesh, or an empty handle if it has none.
    const MicromapHandle& get(const uint32_t mesh_id) const;

    // How many triangles a mesh's micromap covers, and at which level and format.
    const vk::MicromapUsageEXT& get_usage(const uint32_t mesh_id) const;

    void clear();

    void properties(Properties& props);

  private:
    void ensure_pipeline(const SlangCompositionHandle& material_system_composition);

    const ShaderCompileContextHandle compile_context;
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;

    MicromapBuilder builder;

    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    Versioned<SlangProgramEntryPoint> entry_point;
    Versioned<Pipeline> pipeline;
    Versioned<ShaderObject> params;

    ShaderObjectAllocatorHandle fallback_obj_allocator;

    BufferHandle job_buffer;
    BufferHandle data_buffer;
    BufferHandle triangle_buffer;

    struct Entry {
        MicromapHandle micromap;
        vk::MicromapUsageEXT usage;
    };
    std::unordered_map<uint32_t, Entry> entries;

    vk::DeviceSize data_size = 0;
    uint32_t triangle_count = 0;
};

} // namespace merian
